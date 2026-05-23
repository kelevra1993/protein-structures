import torch
import numpy as np

from typing import Optional, Tuple
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention

"""
Todo Note to self :
  Rename c_m -> msa_embedding
  Rename c_z -> pair_representation_embedding
  Rename c -> head_embedding_dimension
  Rename N_head -> number_heads
  Rename m -> msa_representation
  Rename z -> pair_representation
  Rename N_seq -> number_sequences
"""


class MSARowAttentionWithPairBias(nn.Module):
    """
    Implements the MSA Row-wise Gated Self-Attention with Pair Bias for the AlphaFold II Evoformer block.
    This module performs self-attention over the sequence (residue) dimension of the MSA representation 
    independently for each MSA sequence (row-wise).
    """

    def __init__(self, msa_embedding: int, pair_representation_embedding: int, head_embedding_dimension: int,
                 number_heads: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the MSARowAttentionWithPairBias module.
        
        Configures layer normalizations for both MSA and pair representations, a linear embedding to 
        transform the pair representation into an attention bias, and the underlying MultiHeadAttention 
        component scoped to the residue dimension (attention_dimension=-2).

        Args:
            msa_embedding (int): The feature dimension of the MSA representation tensor.
            pair_representation_embedding (int): The feature dimension of the pair representation tensor.
            head_embedding_dimension (int): The dimensionality of each individual attention head.
            number_heads (int): The total number of attention heads to employ.
            device (torch.device): The computational device (e.g., CPU, CUDA) for parameter allocation.
            dtype (torch.dtype): The numerical precision data type for the parameters.
        """
        super().__init__()

        self.msa_embedding = msa_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.head_embedding_dimension = head_embedding_dimension
        self.number_heads = number_heads
        self.device = device
        self.dtype = dtype

        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.msa_embedding,
                                                                device=self.device, dtype=self.dtype)
        self.pair_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.pair_representation_embedding,
                                                                 device=self.device, dtype=self.dtype)

        self.pair_representation_embedder = nn.Linear(in_features=self.pair_representation_embedding,
                                                      out_features=self.number_heads, bias=False,
                                                      device=self.device, dtype=self.dtype)

        # Since this is row wise attention, the attention dimension is -2,
        # in our *, number_sequences, number_residues, number_residues, msa_embedding
        # Our residue rows are the first number_residue dimension, and the column's are the second one
        # *, number_sequences, row_residues, column_residue, msa_embedding
        # -2 since for row-wise attention the column_residue is the one iterated over by attention.
        self.multi_head_attention = MultiHeadAttention(input_dimension=self.msa_embedding,
                                                       head_embedding_dimension=self.head_embedding_dimension,
                                                       number_heads=self.number_heads,
                                                       attention_dimension=-2,
                                                       use_gating=True,
                                                       use_global_attention=False,
                                                       use_embedding_bias=False,
                                                       device=self.device,
                                                       dtype=self.dtype)

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the MSA Row-wise Attention with Pair Bias.

        This method applies row-wise self-attention across the MSA representation. It computes an
        attention bias directly from the 2D pair representation to integrate spatial and structural 
        hypotheses into the evolutionary feature extraction process.

        Args:
            msa_representation (torch.Tensor): The input MSA features.
                Shape: (..., number_clusters, number_residues, msa_embedding_dimension)
            pair_representation (torch.Tensor): The input pair features used to derive the attention bias.
                Shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated MSA representation after applying row-wise attention and gating.
                Shape: (..., number_clusters, number_residues, msa_embedding_dimension)
        """
        normalized_msa_representation = self.msa_representation_layer_normalizer(msa_representation)

        # Creation of bias
        # Here we just normalise it, turn the pair_representation_embedding to a "number_heads" for broadcasting
        # We also have to move the dimension since our tensors in pytorch are of shape :
        # *, number_heads, number_residues, number_residues
        normalized_pair_representaion = self.pair_representation_layer_normalizer(pair_representation)
        bias_tensor = self.pair_representation_embedder(normalized_pair_representaion).movedim(source=-1,
                                                                                               destination=-3)

        output_tensor = self.multi_head_attention.forward(input_tensor=normalized_msa_representation,
                                                          bias_tensor=bias_tensor)

        return output_tensor


class MSAColumnAttention(nn.Module):

    def __init__(self, msa_embedding: int, head_embedding_dimension: int,
                 number_heads: int, device: torch.device, dtype: torch.dtype):
        """"""
        super().__init__()

        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=msa_embedding)
        self.multi_head_attention = MultiHeadAttention(input_dimension=msa_embedding,
                                                       head_embedding_dimension=head_embedding_dimension,
                                                       number_heads=number_heads,
                                                       attention_dimension=-3,
                                                       use_gating=True,
                                                       use_global_attention=False,
                                                       use_embedding_bias=False,
                                                       device=device,
                                                       dtype=dtype)

    def forward(self, msa_representation: torch.Tensor) -> torch.Tensor:
        normalized_msa_representation = self.msa_representation_layer_normalizer(msa_representation)
        output_tensor = self.multi_head_attention.forward(input_tensor=normalized_msa_representation)

        return output_tensor


class MSATransition(nn.Module):

    def __init__(self, msa_embedding, channel_scaler=4):
        super().__init__()

        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=msa_embedding)
        self.first_msa_representation_embedder = nn.Linear(in_features=msa_embedding,
                                                           out_features=channel_scaler * msa_embedding)
        self.relu = nn.ReLU()
        self.second_msa_representation_embedder = nn.Linear(in_features=channel_scaler * msa_embedding,
                                                            out_features=msa_embedding)
        self.sequential = nn.Sequential(
            self.msa_representation_layer_normalizer,
            self.first_msa_representation_embedder,
            self.relu,
            self.second_msa_representation_embedder,
        )

    def forward(self, msa_representation):
        output = self.sequential(msa_representation)

        return output


class OuterProductMean(nn.Module):

    def __init__(self, msa_embedding, pair_representation_embedding, intermediate_embedding):
        super().__init__()
        self.intermediate_embedding = intermediate_embedding
        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=msa_embedding)
        self.first_msa_representation_embedder = nn.Linear(in_features=msa_embedding,
                                                           out_features=intermediate_embedding)
        self.second_msa_representation_embedder = nn.Linear(in_features=msa_embedding,
                                                            out_features=intermediate_embedding)
        self.linear_pair_represenation_embedder = nn.Linear(in_features=intermediate_embedding * intermediate_embedding,
                                                            out_features=pair_representation_embedding)

    def forward(self, msa_representation):
        number_sequences = msa_representation.shape[-3]

        normalized_msa_representation = self.msa_representation_layer_normalizer(msa_representation)
        left_matrix = self.first_msa_representation_embedder(normalized_msa_representation)
        right_matrix = self.second_msa_representation_embedder(normalized_msa_representation)

        output_matrix = torch.einsum('...sic,...sjd->...ijcd', left_matrix, right_matrix)
        flattened_output_matrix = torch.flatten(input=output_matrix, start_dim=-2, end_dim=-1)

        pair_representation = self.linear_pair_represenation_embedder(flattened_output_matrix) / number_sequences

        return pair_representation
