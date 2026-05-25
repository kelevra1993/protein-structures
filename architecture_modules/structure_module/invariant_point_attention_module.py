import torch
import math
from torch import nn

from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector

"""
TODO : Note to self
Rename c_s -> single_representation_embedding
Rename c_z -> pair_representation_embedding
Rename n_query_points -> number_query_points
Rename n_point_values -> number_value_points
Rename N_head -> number_heads
Rename c -> head_embedding_dimension
Rename s -> single_representation
Rename z -> pair_representation
Rename q -> query_tensor
Rename k -> key_tensor
Rename qp-> query_point_tensor
Rename kp-> key_point_tensor
Rename T -> transformation_matrix
Rename warp_3d_point -> apply_transformation_on_vector
Rename N_res -> number_residues
"""


class InvariantPointAttention(nn.Module):
    """
    Implements the Invariant Point Attention (IPA) mechanism, a central component of the AlphaFold II Structure Module.

    IPA is designed to be invariant to global rotations and translations of the protein structure. It achieves this
    by performing attention in the local coordinate frames of each residue. This module combines three distinct
    sources of information:
    1.  Scalar Attention: Standard multi-head attention on the single representation (residue-level features).
    2.  Pair Bias: Incorporates spatial and relational information from the pair representation.
    3.  Geometric Attention: Computes interactions based on the 3D distances between points projected from
        each residue into global space and then transformed back into local frames.

    In the context of the global project, IPA is used iteratively within the Structure Module to refine the
    3D coordinates of the protein backbone by allowing residues to "communicate" their relative spatial
    orientations and distances in a physically meaningful, invariant way.

    Attributes:
        single_representation_embedding (int): The feature dimension of the input single representation.
        pair_representation_embedding (int): The feature dimension of the input pair representation.
        number_query_points (int): The number of points projected per head for query-key distance calculations.
        number_value_points (int): The number of points projected per head for value aggregation.
        number_heads (int): The total number of attention heads.
        head_embedding_dimension (int): The dimensionality of each attention head.
        device (torch.device): The computational device on which the module's parameters reside.
        dtype (torch.dtype): The numerical precision data type of the module's parameters.
    """

    def __init__(self, single_representation_embedding: int, pair_representation_embedding: int,
                 number_query_points: int = 4, number_value_points: int = 8, number_heads: int = 12,
                 head_embedding_dimension: int = 16, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the InvariantPointAttention module with the specified architectural dimensions.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            pair_representation_embedding (int): Feature dimension of the pair representation.
            number_query_points (int): Number of geometric query points per head. Defaults to 4.
            number_value_points (int): Number of geometric value points per head. Defaults to 8.
            number_heads (int): Number of attention heads. Defaults to 12.
            head_embedding_dimension (int): Hidden dimension per head. Defaults to 16.
            device (torch.device, optional): Device for tensor allocation.
            dtype (torch.dtype, optional): Data type for tensors.
        """
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_heads = number_heads
        self.head_embedding_dimension = head_embedding_dimension
        self.device = device
        self.dtype = dtype

        # Here alpha fold's implementation did not use bias but we will use Bias
        self.linear_k = nn.Linear(in_features=self.single_representation_embedding,
                                  out_features=self.head_embedding_dimension * self.number_heads,
                                  bias=True, device=self.device, dtype=self.dtype)
        self.linear_q = nn.Linear(in_features=self.single_representation_embedding,
                                  out_features=self.head_embedding_dimension * self.number_heads,
                                  bias=True, device=self.device, dtype=self.dtype)
        self.linear_v = nn.Linear(in_features=self.single_representation_embedding,
                                  out_features=self.head_embedding_dimension * self.number_heads,
                                  bias=True, device=self.device, dtype=self.dtype)

        self.linear_k_points = nn.Linear(in_features=self.single_representation_embedding,
                                         out_features=self.number_heads * self.number_query_points * 3,
                                         bias=True, device=self.device, dtype=self.dtype)
        self.linear_q_points = nn.Linear(in_features=self.single_representation_embedding,
                                         out_features=self.number_heads * self.number_query_points * 3,
                                         bias=True, device=self.device, dtype=self.dtype)
        self.linear_v_points = nn.Linear(in_features=self.single_representation_embedding,
                                         out_features=self.number_heads * self.number_value_points * 3,
                                         bias=True, device=self.device, dtype=self.dtype)

        self.linear_b = nn.Linear(in_features=self.pair_representation_embedding,
                                  out_features=self.number_heads,
                                  bias=True, device=self.device, dtype=self.dtype)

        self.linear_out = nn.Linear(
            in_features=self.number_heads * (self.head_embedding_dimension +
                                             self.pair_representation_embedding +
                                             self.number_value_points * 4),
            out_features=self.single_representation_embedding,
            bias=True, device=self.device, dtype=self.dtype)

        self.head_weights = nn.Parameter(torch.zeros((self.number_heads,), device=self.device, dtype=self.dtype))
        self.softplus = nn.Softplus()

    def separate_key_query_value_heads(self, single_representation: torch.Tensor):
        """
        Projects the input single representation into scalar and point-based Query, Key, and Value spaces.

        This method applies linear transformations to the single representation to create the necessary
        components for both standard and geometric attention. It then reshapes these projections to
        separate the attention heads and, for point-based features, the 3D coordinates.

        Args:
            single_representation (torch.Tensor): The residue-level features.
                Shape: `(..., number_residues, single_representation_embedding)`.

        Returns:
            tuple[torch.Tensor, ...]: A tuple of six tensors:
                - query_tensor: Scalar queries. Shape `(..., number_heads, number_residues, head_embedding_dimension)`.
                - key_tensor: Scalar keys. Shape `(..., number_heads, number_residues, head_embedding_dimension)`.
                - value_tensor: Scalar values. Shape `(..., number_heads, number_residues, head_embedding_dimension)`.
                - query_point_tensor: Geometric query points. Shape `(..., number_heads, number_query_points, number_residues, 3)`.
                - key_point_tensor: Geometric key points. Shape `(..., number_heads, number_query_points, number_residues, 3)`.
                - value_point_tensor: Geometric value points. Shape `(..., number_heads, number_value_points, number_residues, 3)`.
        """
        head_embedding_dimension = self.head_embedding_dimension
        number_heads = self.number_heads
        number_query_points = self.number_query_points
        number_value_points = self.number_value_points

        layers = [self.linear_q, self.linear_k, self.linear_v, self.linear_q_points, self.linear_k_points,
                  self.linear_v_points]
        embeddings = [layer(single_representation) for layer in layers]

        # Solution proposed by Kilian Mandon
        shape_adds = [
            (number_heads, head_embedding_dimension),
            (number_heads, head_embedding_dimension),
            (number_heads, head_embedding_dimension),
            (3, number_heads, number_query_points),
            (3, number_heads, number_query_points),
            (3, number_heads, number_value_points)
        ]

        out_shapes = [out.shape[:-1] + shape_add for out, shape_add in zip(embeddings, shape_adds)]
        embeddings = [out.view(out_shape) for out, out_shape in zip(embeddings, out_shapes)]
        for i in range(3):
            embeddings[i] = embeddings[i].movedim(-3, -2)
        for i in range(3, 6):
            embeddings[i] = embeddings[i].movedim(-3, -1).movedim(-4, -2)

        # TODO Decide on which implementation makes sense.
        # # Personal implementation
        # for i in range(3):
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=self.head_embedding_dimension, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)
        #     print_shape(name="After Reshape Tensor Points", tensor=embeddings[i])
        #
        # points = [number_query_points, number_query_points, number_value_points]
        # for i in range(3,6):
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=points[i%3]*3, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)
        #     embedding_chunks = torch.split(embeddings[i], split_size_or_sections=3, dim=-1)
        #     embeddings[i] = torch.stack(embedding_chunks, dim=-3)

        return embeddings

    def compute_attention_scores(self, query_tensor: torch.Tensor, key_tensor: torch.Tensor,
                                 query_point_tensor: torch.Tensor, key_point_tensor: torch.Tensor,
                                 pair_representation: torch.Tensor, transformation_matrix: torch.Tensor):
        """
        Calculates normalized attention scores by integrating scalar, pairwise, and geometric components.

        The total attention score is a sum of:
        - Scaled dot-product of scalar queries and keys.
        - Relational bias from the pair representation projected into the head space.
        - Geometric penalty proportional to the squared 3D distance between query and key points
          transformed into the global frame using the provided backbone transformations.

        Args:
            query_tensor (torch.Tensor): Scalar query embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            key_tensor (torch.Tensor): Scalar key embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            query_point_tensor (torch.Tensor): Geometric query points in local residue frames.
                Shape: `(..., number_heads, number_query_points, number_residues, 3)`.
            key_point_tensor (torch.Tensor): Geometric key points in local residue frames.
                Shape: `(..., number_heads, number_query_points, number_residues, 3)`.
            pair_representation (torch.Tensor): Pair representation used as relational attention bias.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            transformation_matrix (torch.Tensor): Rigid backbone transformations (4x4 matrices).
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            torch.Tensor: Normalized attention scores after applying softmax along the key dimension.
                Shape: `(..., number_heads, number_residues, number_residues)`.
        """

        wc = math.sqrt(2 / (9 * self.number_query_points))
        wl = math.sqrt(1 / 3)

        # Standard dot-product attention
        scaler = math.sqrt(1 / self.head_embedding_dimension)
        attention = torch.matmul(query_tensor, torch.transpose(key_tensor, dim0=-1, dim1=-2)) * scaler

        # Calculate attention bias from pair representation
        bias = self.linear_b(pair_representation)
        bias = bias.movedim(source=-1, destination=-3)

        # Reshape transformation matrix for broadcasting with point coordinates
        transformation_matrix = transformation_matrix.unsqueeze(dim=-4).unsqueeze(dim=-4)

        # Calculate geometric attention weights
        scaled_head_weights = self.softplus(self.head_weights) * (wc / 2)
        scaled_head_weights = scaled_head_weights.view((-1, 1, 1))

        # Personal Implementation
        # global_query = apply_transformation_on_vector(transformation_matrix=transformation_matrix, vector=query_point_tensor)
        # global_key = apply_transformation_on_vector(transformation_matrix=transformation_matrix, vector=key_point_tensor)
        # key_query_distance_squared = torch.linalg.vector_norm(global_key.unsqueeze(-3) - global_query.unsqueeze(-2), dim=-1, keepdim=True)**2
        # proposal that might need to be checked to be properly understood especially the outer product

        # Transform points to the global frame and compute squared distances
        global_query = apply_transformation_on_vector(transformation_matrix=transformation_matrix,
                                                      vector=query_point_tensor).unsqueeze(-2)
        global_key = apply_transformation_on_vector(transformation_matrix=transformation_matrix,
                                                    vector=key_point_tensor).unsqueeze(-3)
        key_query_distance_squared = torch.linalg.vector_norm((global_query - global_key), dim=-1) ** 2
        sum_key_query_distances_squared = torch.sum(key_query_distance_squared, dim=-3)

        # Combine all components and apply softmax
        attention_scores = torch.softmax(
            wl * (attention + bias - scaled_head_weights * sum_key_query_distances_squared), dim=-1)

        return attention_scores

    def compute_outputs(self, attention_scores: torch.Tensor, pair_representation: torch.Tensor,
                        value_tensor: torch.Tensor, value_point_tensor: torch.Tensor,
                        transformation_matrix: torch.Tensor):
        """
        Aggregates scalar, geometric, and pairwise information weighted by the attention scores.

        This method produces the final components that will be concatenated and projected to update the
        single representation. It handles:
        - Weighted sum of scalar values.
        - Weighted sum of pair representations.
        - Weighted sum of point values in the global frame, which are then transformed back into local frames.

        Args:
            attention_scores (torch.Tensor): Normalized attention weights.
                Shape: `(..., number_heads, number_residues, number_residues)`.
            pair_representation (torch.Tensor): The input pair features.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            value_tensor (torch.Tensor): Scalar value embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            value_point_tensor (torch.Tensor): Geometric value points in local frames.
                Shape: `(..., number_heads, number_value_points, number_residues, 3)`.
            transformation_matrix (torch.Tensor): Backbone transformation matrices.
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            tuple[torch.Tensor, ...]: A tuple of four aggregated tensors:
                - value_output: Aggregated scalar values. Shape `(..., number_residues, number_heads * head_embedding_dimension)`.
                - value_point_output: Aggregated points in local frames. Shape `(..., number_residues, number_heads * 3 * number_value_points)`.
                - value_point_output_norm: Norms of the local aggregated points. Shape `(..., number_residues, number_heads * number_value_points)`.
                - pair_representation_output: Aggregated pair features. Shape `(..., number_residues, number_heads * pair_representation_embedding)`.
        """

        # Scalar value aggregation (classic attention)
        value_output = torch.einsum('...hij,...hjc->...hic', attention_scores, value_tensor)
        value_output = value_output.movedim(source=-3, destination=-2).flatten(start_dim=-2)

        # Pairwise representation aggregation
        pair_representation_output = torch.einsum('...hij,...ijc->...hic', attention_scores, pair_representation)
        pair_representation_output = pair_representation_output.movedim(
            source=-3, destination=-2).flatten(start_dim=-2, end_dim=-1)

        # 3D Point value aggregation
        transformation_matrix = transformation_matrix.unsqueeze(dim=-4).unsqueeze(dim=-4)
        global_value_point_output = apply_transformation_on_vector(transformation_matrix=transformation_matrix,
                                                            vector=value_point_tensor)
        scaled_global_value_point = torch.einsum('...Bij,...BNjk->...BNik',
                                                 attention_scores, global_value_point_output)

        # Invert transformation to bring aggregated points back to the local frames
        value_point_output = apply_transformation_on_vector(
            transformation_matrix=invert_4x4_transform_matrix(transformation_matrix),
            vector=scaled_global_value_point)
        value_point_output = torch.einsum('...hpic->...ichp', value_point_output)
        value_point_output_norm = torch.linalg.vector_norm(value_point_output, dim=-3, keepdim=True)

        value_point_output = value_point_output.flatten(start_dim=-3)
        value_point_output_norm = value_point_output_norm.flatten(start_dim=-3)

        return value_output, value_point_output, value_point_output_norm, pair_representation_output

    def forward(self, single_representation: torch.Tensor,
                pair_representation: torch.Tensor,
                transformation_matrix: torch.Tensor):
        """
        Executes the full forward pass of the Invariant Point Attention module.

        This method orchestrates the IPA pipeline: projecting inputs into heads, computing scalar
        and geometric attention scores, aggregating the weighted outputs, and finally projecting
        the concatenated results back to the single representation dimension.

        Args:
            single_representation (torch.Tensor): The input residue-level features.
                Shape: `(..., number_residues, single_representation_embedding)`.
            pair_representation (torch.Tensor): The input pairwise features.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            transformation_matrix (torch.Tensor): The current rigid body backbone transformations.
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            torch.Tensor: The updated single representation.
                Shape: `(..., number_residues, single_representation_embedding)`.
        """

        # Step 1: Project inputs and separate into heads
        query_tensor, key_tensor, value_tensor, query_point_tensor, key_point_tensor, value_point_tensor = \
            self.separate_key_query_value_heads(single_representation=single_representation)

        # Step 2: Compute attention scores
        attention_scores = self.compute_attention_scores(
            query_tensor=query_tensor, key_tensor=key_tensor,
            query_point_tensor=query_point_tensor, key_point_tensor=key_point_tensor,
            pair_representation=pair_representation, transformation_matrix=transformation_matrix)

        # Step 3: Aggregate values and compute outputs
        v_out, vp_out, vp_out_norm, pairwise_out = self.compute_outputs(
            attention_scores=attention_scores, pair_representation=pair_representation,
            value_tensor=value_tensor, value_point_tensor=value_point_tensor,
            transformation_matrix=transformation_matrix)

        # Step 4: Concatenate outputs and project to final single representation dimension
        output_tensor = self.linear_out(torch.cat(tensors=(v_out, vp_out, vp_out_norm, pairwise_out), dim=-1))

        return output_tensor
