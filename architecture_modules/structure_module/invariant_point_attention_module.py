import torch
import math
from torch import nn

from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector




class InvariantPointAttention(nn.Module):
    """
    Implements the Invariant Point Attention (IPA) mechanism from the AlphaFold II Structure Module.

    IPA is a specialized attention mechanism that is invariant to global rotations and translations.
    It combines standard attention on the single and pair representations with a 3D geometric attention
    that operates on point coordinates in the local frames of each residue. This allows the model
    to reason about the physical distances and orientations between residues in 3D space.

    It is used within the Structure Module's iterative updates to refine the predicted protein 
    backbone coordinates by integrating local geometric information with sequence and pairwise features.

    Attributes:
        single_representation_embedding (int): Dimension of the input single representation.
        pair_representation_embedding (int): Dimension of the input pair representation.
        number_query_points (int): Number of query points projected into 3D space per head.
        number_value_points (int): Number of value points projected into 3D space per head.
        number_heads (int): Total number of attention heads.
        head_embedding_dimension (int): Dimension of each individual attention head.
    """

    def __init__(self, single_representation_embedding: int, pair_representation_embedding: int,
                 number_query_points: int = 4, number_value_points: int = 8, number_heads: int = 12,
                 head_embedding_dimension: int = 16, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the InvariantPointAttention module.

        Args:
            single_representation_embedding (int): Dimension of the input single representation.
            pair_representation_embedding (int): Dimension of the input pair representation.
            number_query_points (int): Number of query points per head. Defaults to 4.
            number_value_points (int): Number of value points per head. Defaults to 8.
            number_heads (int): Number of attention heads. Defaults to 12.
            head_embedding_dimension (int): Dimension of each attention head. Defaults to 16.
            device (torch.device): Computational device for parameter allocation.
            dtype (torch.dtype): Numerical precision data type.
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
        Projects the single representation and splits it into multiple heads for IPA.

        This method produces standard attention embeddings (q, k, v) and 3D point
        embeddings (qp, kp, vp) used for invariant point attention.

        Args:
            single_representation (torch.Tensor): Input single representation.
                Shape: `(..., number_residues, single_representation_embedding)`.

        Returns:
            tuple[torch.Tensor, ...]: A tuple containing:
                - query_tensor: `(..., number_heads, number_residues, head_embedding_dimension)`
                - key_tensor: `(..., number_heads, number_residues, head_embedding_dimension)`
                - value_tensor: `(..., number_heads, number_residues, head_embedding_dimension)`
                - query_point_tensor: `(..., number_heads, number_query_points, number_residues, 3)`
                - key_point_tensor: `(..., number_heads, number_query_points, number_residues, 3)`
                - value_point_tensor: `(..., number_heads, number_value_points, number_residues, 3)`
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
        Computes the attention scores by combining scalar attention, pair bias, and 3D point distances.
        It calculates the standard dot-product attention, adds a bias derived from the pair representation,
        and subtracts weighted squared distances between transformed 3D points.

        Args:
            query_tensor (torch.Tensor): Query embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            key_tensor (torch.Tensor): Key embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            query_point_tensor (torch.Tensor): Query point embeddings in local frames.
                Shape: `(..., number_heads, number_query_points, number_residues, 3)`.
            key_point_tensor (torch.Tensor): Key point embeddings in local frames.
                Shape: `(..., number_heads, number_query_points, number_residues, 3)`.
            pair_representation (torch.Tensor): Pair representation used as attention bias.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            transformation_matrix (torch.Tensor): Backbone transformation matrices (rigid groups).
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            torch.Tensor: Normalized attention scores.
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
        Computes the final IPA outputs by applying attention to value representations.

        Aggregates scalar values, 3D point values, and the pair representation 
        using the calculated attention scores.

        Args:
            attention_scores (torch.Tensor): Normalized attention scores.
                Shape: `(..., number_heads, number_residues, number_residues)`.
            pair_representation (torch.Tensor): Pair representation.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            value_tensor (torch.Tensor): Value embeddings.
                Shape: `(..., number_heads, number_residues, head_embedding_dimension)`.
            value_point_tensor (torch.Tensor): Value point embeddings in local frames.
                Shape: `(..., number_heads, number_value_points, number_residues, 3)`.
            transformation_matrix (torch.Tensor): Backbone transformation matrices.
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            tuple[torch.Tensor, ...]: A tuple containing:
                - v_out: Aggregated scalar values. Shape: `(..., number_residues, number_heads * head_embedding_dimension)`.
                - vp_out: Aggregated and re-projected 3D point values. Shape: `(..., number_residues, number_heads * 3 * number_value_points)`.
                - vp_out_norm: Norms of aggregated 3D point values. Shape: `(..., number_residues, number_heads * number_value_points)`.
                - pairwise_out: Aggregated pair representations. Shape: `(..., number_residues, number_heads * pair_representation_embedding)`.
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
        Executes the forward pass of the Invariant Point Attention module.

        Args:
            single_representation (torch.Tensor): Input single representation.
                Shape: `(..., number_residues, single_representation_embedding)`.
            pair_representation (torch.Tensor): Input pair representation.
                Shape: `(..., number_residues, number_residues, pair_representation_embedding)`.
            transformation_matrix (torch.Tensor): Current backbone transformation matrices.
                Shape: `(..., number_residues, 4, 4)`.

        Returns:
            torch.Tensor: Updated single representation.
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
