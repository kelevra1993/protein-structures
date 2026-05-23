import torch
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention


class TriangleMultiplication(nn.Module):
    """
    Implements the Triangle Multiplication module for the AlphaFold II Evoformer block.
    """

    def __init__(self, pair_representation_embedding: int, multiplication_type: str, embedding_dimension: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the TriangleMultiplication module.

        Args:
            pair_representation_embedding (int): The feature dimension of the input pair representation.
            multiplication_type (str): The type of multiplication to perform ('outgoing' or 'incoming').
            embedding_dimension (int): The reduced hidden dimension used during the triangle update computation.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """
        super().__init__()

        self.pair_representation_embedding = pair_representation_embedding
        self.multiplication_type = multiplication_type
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.dtype = dtype

        self.pair_representation_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.pair_representation_embedding, device=self.device, dtype=self.dtype)
        self.output_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.embedding_dimension, device=self.device, dtype=self.dtype)

        # For Left Representation
        self.first_pair_representation_embedder = nn.Linear(
            in_features=self.pair_representation_embedding, out_features=self.embedding_dimension,
            device=self.device, dtype=self.dtype)
        self.first_gating_embedder = nn.Linear(
            in_features=self.pair_representation_embedding, out_features=self.embedding_dimension,
            device=self.device, dtype=self.dtype)

        # For Right Representation
        self.second_pair_representation_embedder = nn.Linear(
            in_features=self.pair_representation_embedding, out_features=self.embedding_dimension,
            device=self.device, dtype=self.dtype)
        self.second_gating_embedder = nn.Linear(
            in_features=self.pair_representation_embedding, out_features=self.embedding_dimension,
            device=self.device, dtype=self.dtype)

        self.third_pair_representation_embedder = nn.Linear(
            in_features=self.pair_representation_embedding, out_features=self.pair_representation_embedding,
            device=self.device, dtype=self.dtype)
        self.output_embedder = nn.Linear(
            in_features=self.embedding_dimension, out_features=self.pair_representation_embedding,
            device=self.device, dtype=self.dtype)

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the Triangle Multiplication module.

        Args:
            pair_representation (torch.Tensor): The input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated pair representation after the triangle multiplication operation.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)
        """
        normalized_pair_representation = self.pair_representation_layer_normalizer(pair_representation)

        left_representation = (torch.sigmoid(self.first_gating_embedder(normalized_pair_representation)) *
                               self.first_pair_representation_embedder(normalized_pair_representation))

        right_representation = (torch.sigmoid(self.second_gating_embedder(normalized_pair_representation)) *
                                self.second_pair_representation_embedder(normalized_pair_representation))

        gate_tensor = torch.sigmoid(self.third_pair_representation_embedder(normalized_pair_representation))

        if self.multiplication_type == 'outgoing':
            applied_triangle_multiplication_tensor = torch.einsum('...ikc,...jkc->...ijc', left_representation,
                                                                  right_representation)
        else:
            applied_triangle_multiplication_tensor = torch.einsum('...kic,...kjc->...ijc', left_representation,
                                                                  right_representation)

        output_tensor = gate_tensor * self.output_embedder(
            self.output_layer_normalizer(applied_triangle_multiplication_tensor))

        return output_tensor


class TriangleAttention(nn.Module):
    """
    Implements the Triangle Attention module for the AlphaFold II Evoformer block.
    """

    def __init__(self, pair_representation_embedding: int, node_type: str, head_embedding_dimension: int,
                 number_heads: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the TriangleAttention module.

        Args:
            pair_representation_embedding (int): The feature dimension of the input pair representation.
            node_type (str): Specifies the attention graph structure ("starting_node" or "ending_node").
            head_embedding_dimension (int): The dimensionality of each individual attention head.
            number_heads (int): The total number of attention heads to employ.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """
        super().__init__()
        if node_type not in {'starting_node', 'ending_node'}:
            raise ValueError(f'node_type must be either "starting_node" or "ending_node" but is {node_type}')

        self.pair_representation_embedding = pair_representation_embedding
        self.node_type = node_type
        self.head_embedding_dimension = head_embedding_dimension
        self.number_heads = number_heads
        self.device = device
        self.dtype = dtype

        self.pair_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.pair_representation_embedding,
                                                                 device=self.device, dtype=self.dtype)

        self.multi_head_attention = MultiHeadAttention(input_dimension=self.pair_representation_embedding,
                                                       head_embedding_dimension=self.head_embedding_dimension,
                                                       number_heads=self.number_heads,
                                                       attention_dimension=-2 if self.node_type == "starting_node" else -3,
                                                       use_gating=True,
                                                       use_global_attention=False,
                                                       use_embedding_bias=False,
                                                       device=self.device,
                                                       dtype=self.dtype)

        self.pair_representation_embedder = nn.Linear(in_features=self.pair_representation_embedding,
                                                      out_features=self.number_heads, bias=False,
                                                      device=self.device, dtype=self.dtype)

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the Triangle Attention module.

        Args:
            pair_representation (torch.Tensor): The input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated pair representation after triangle attention.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)
        """
        normalized_pair_representation = self.pair_representation_layer_normalizer(pair_representation)
        bias_tensor = self.pair_representation_embedder(normalized_pair_representation).movedim(-1, -3)

        if self.node_type == "starting_node":
            output_tensor = self.multi_head_attention(input_tensor=normalized_pair_representation,
                                                      bias_tensor=bias_tensor)
        else:
            output_tensor = self.multi_head_attention(input_tensor=normalized_pair_representation,
                                                      bias_tensor=torch.transpose(input=bias_tensor, dim0=-2, dim1=-1))

        return output_tensor


class PairTransition(nn.Module):
    """
    Implements the Pair Transition layer for the AlphaFold II Evoformer block.

    This is a 2-layer feed-forward network applied independently to each position 
    (i, j) in the pair representation. It increases the channel dimension by a 
    scaling factor and then projects it back to the original dimension.
    """

    def __init__(self, pair_representation_embedding: int, channel_scaler: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the PairTransition module.

        Args:
            pair_representation_embedding (int): The feature dimension of the input pair representation.
            channel_scaler (int): The scaling factor for the hidden layer dimension.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """
        super().__init__()

        self.pair_representation_embedding = pair_representation_embedding
        self.channel_scaler = channel_scaler
        self.device = device
        self.dtype = dtype

        self.pair_representation_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.pair_representation_embedding,
            device=self.device, dtype=self.dtype
        )

        self.first_pair_representation_embedder = nn.Linear(
            in_features=self.pair_representation_embedding,
            out_features=self.channel_scaler * self.pair_representation_embedding,
            device=self.device, dtype=self.dtype
        )

        self.relu = nn.ReLU()

        self.second_pair_representation_embedder = nn.Linear(
            in_features=self.channel_scaler * self.pair_representation_embedding,
            out_features=self.pair_representation_embedding,
            device=self.device, dtype=self.dtype
        )

        # Use sequential here because it is more straight forward and easier to use
        self.sequential = nn.Sequential(
            self.pair_representation_layer_normalizer,
            self.first_pair_representation_embedder,
            self.relu,
            self.second_pair_representation_embedder,
        )

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the Pair Transition layer.

        Args:
            pair_representation (torch.Tensor): The input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated pair representation after the transition.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)
        """
        output_tensor = self.sequential(pair_representation)
        return output_tensor


class PairStack(nn.Module):
    """
    Implements a single block of the Pair Stack for the AlphaFold II Evoformer.

    This module orchestrates the sequential application of triangle updates 
    (outgoing and incoming), triangle attention (starting and ending nodes), 
    and a final transition layer. Each sub-module includes a residual connection.
    """

    def __init__(self, pair_representation_dimension: int,
                 head_embedding_dimension: int,triangle_multiplication_embedding: int,
                 number_heads: int, channel_scaler: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the PairStack module.

        Args:
            pair_representation_dimension (int): The feature dimension of the input pair representation.
            head_embedding_dimension (int): The dimensionality of each individual attention head.
            triangle_multiplication_embedding (int): The reduced hidden dimension used during the triangle updates.
            number_heads (int): The total number of attention heads.
            channel_scaler (int): The scaling factor for the PairTransition hidden layer.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """
        super().__init__()

        self.pair_representation_dimension = pair_representation_dimension
        self.head_embedding_dimension = head_embedding_dimension
        self.triangle_multiplication_embedding = triangle_multiplication_embedding
        self.number_heads = number_heads
        self.channel_scaler = channel_scaler
        self.device = device
        self.dtype = dtype

        # First step after outer product mean
        self.triangle_update_outgoing_edges = TriangleMultiplication(
            pair_representation_embedding=self.pair_representation_dimension,
            multiplication_type="outgoing",
            embedding_dimension=self.triangle_multiplication_embedding,
            device=self.device, dtype=self.dtype)

        # Second step after outer product mean
        self.triangle_update_incoming_edges = TriangleMultiplication(
            pair_representation_embedding=self.pair_representation_dimension,
            multiplication_type="incoming",
            embedding_dimension=self.triangle_multiplication_embedding,
            device=self.device, dtype=self.dtype)

        # Third step after outer product mean
        self.triangle_attention_starting_nodes = TriangleAttention(
            pair_representation_embedding=self.pair_representation_dimension,
            node_type="starting_node",
            head_embedding_dimension=self.head_embedding_dimension,
            number_heads=self.number_heads,
            device=self.device, dtype=self.dtype)

        # Fourth step after outer product mean
        self.triangle_attention_ending_nodes = TriangleAttention(
            pair_representation_embedding=self.pair_representation_dimension,
            node_type="ending_node",
            head_embedding_dimension=self.head_embedding_dimension,
            number_heads=self.number_heads,
            device=self.device, dtype=self.dtype)

        # Last step after outer product mean
        self.pair_transition = PairTransition(
            pair_representation_embedding=self.pair_representation_dimension,
            channel_scaler=self.channel_scaler,
            device=self.device, dtype=self.dtype)

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the Pair Stack block.

        Args:
            pair_representation (torch.Tensor): The input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated pair representation after processing all sub-modules.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)
        """

        pair_representation = pair_representation + self.triangle_update_outgoing_edges(pair_representation)
        pair_representation = pair_representation + self.triangle_update_incoming_edges(pair_representation)

        pair_representation = pair_representation + self.triangle_attention_starting_nodes(pair_representation)
        pair_representation = pair_representation + self.triangle_attention_ending_nodes(pair_representation)

        output_tensor = pair_representation + self.pair_transition(pair_representation)

        return output_tensor
