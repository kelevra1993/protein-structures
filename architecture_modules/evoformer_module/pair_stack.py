import torch
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention

"""
Todo Note to self :
  Rename c_m -> msa_embedding
  Rename c_z -> pair_representation_embedding
  Rename c -> embedding_dimension
  Rename N_head -> number_heads
  Rename m -> msa_representation
  Rename z -> pair_representation
  Rename N_seq -> number_sequences
  Rename mult_type -> multiplication_type
"""


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
            device (torch.device, optional): The computational device.
            dtype (torch.dtype, optional): The numerical precision data type.
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
