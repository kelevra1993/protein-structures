from torch import nn

from architecture_modules.evoformer_module.msa_stack import (MSARowAttentionWithPairBias, MSAColumnAttention,
                                                             MSATransition, OuterProductMean)
from typing import Tuple
import torch
from architecture_modules.evoformer_module.pair_stack import PairStack


class EvoformerBlock(nn.Module):
    """
    Implements a single block of the AlphaFold II Evoformer architecture.

    The Evoformer block facilitates communication between the MSA representation 
    and the pair representation. It alternates between updating the MSA representation 
    using the pair representation as a bias, and updating the pair representation 
    using an outer product mean of the MSA representation. It includes row/column 
    attention for the MSA, a transition layer for the MSA, and a full PairStack 
    for the pair representation.
    """

    def __init__(self, msa_embedding: int, pair_representation_embedding: int,
                 msa_number_heads: int, msa_head_embedding_dimension: int,
                 pair_number_heads: int, pair_head_embedding_dimension: int,
                 channel_scaler: int,
                 intermediate_embedding: int, triangle_multiplication_embedding: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the EvoformerBlock module.

        Args:
            msa_embedding (int): The feature dimension of the input MSA representation.
            pair_representation_embedding (int): The feature dimension of the input pair representation.
            number_heads (int): The total number of attention heads.
            head_embedding_dimension (int): The dimensionality of each individual attention head.
            channel_scaler (int): The scaling factor for the hidden layer dimensions.
            intermediate_embedding (int): The intermediate dimension for outer product mean.
            triangle_multiplication_embedding (int): The reduced hidden dimension for triangle updates.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """

        super().__init__()
        self.msa_row_wise_attention = MSARowAttentionWithPairBias(
            msa_embedding=msa_embedding,
            pair_representation_embedding=pair_representation_embedding,
            head_embedding_dimension=msa_head_embedding_dimension,
            number_heads=msa_number_heads,
            device=device, dtype=dtype
        )
        self.msa_column_wise_attention = MSAColumnAttention(
            msa_embedding=msa_embedding,
            head_embedding_dimension=msa_head_embedding_dimension,
            number_heads=msa_number_heads,
            device=device, dtype=dtype
        )
        self.msa_transition_embedder = MSATransition(
            msa_embedding=msa_embedding,
            channel_scaler=channel_scaler,
            device=device, dtype=dtype
        )
        self.outer_product_mean = OuterProductMean(
            msa_embedding=msa_embedding,
            pair_representation_embedding=pair_representation_embedding,
            intermediate_embedding=intermediate_embedding,
            device=device, dtype=dtype
        )
        self.pair_stack_embedder = PairStack(
            pair_representation_dimension=pair_representation_embedding,
            head_embedding_dimension=pair_head_embedding_dimension,
            triangle_multiplication_embedding=triangle_multiplication_embedding,
            number_heads=pair_number_heads,
            channel_scaler=channel_scaler,
            device=device, dtype=dtype
        )

    def forward(self, msa_representation: torch.Tensor,
                pair_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the EvoformerBlock.

        Args:
            msa_representation (torch.Tensor): The input MSA features.
                Shape: (..., number_sequences, number_residues, msa_embedding)
            pair_representation (torch.Tensor): The input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_embedding)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated MSA representation 
            and the updated pair representation.
                MSA Shape: (..., number_sequences, number_residues, msa_embedding)
                Pair Shape: (..., number_residues, number_residues, pair_representation_embedding)
        """
        msa_representation += self.msa_row_wise_attention(msa_representation=msa_representation,
                                                          pair_representation=pair_representation)
        msa_representation += self.msa_column_wise_attention(msa_representation)
        msa_representation += self.msa_transition_embedder(msa_representation)

        pair_representation += self.outer_product_mean(msa_representation)
        pair_representation = self.pair_stack_embedder(pair_representation)

        return msa_representation, pair_representation


class EvoformerStack(nn.Module):
    """
    Implements the full Evoformer Stack for the AlphaFold II architecture.

    The stack consists of a sequence of `EvoformerBlock`s. It repeatedly updates 
    both the MSA representation and the pair representation. After processing 
    through all blocks, it extracts the first row of the MSA representation 
    (corresponding to the target sequence) and linearly projects it to form 
    the `single_representation`.
    """

    def __init__(self, msa_embedding: int, pair_representation_embedding: int,
                 number_heads: int, head_embedding_dimension: int, channel_scaler: int,
                 intermediate_embedding: int, triangle_multiplication_embedding: int,
                 number_blocks: int, single_representation_embedding: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the EvoformerStack module.

        Args:
            msa_embedding (int): The feature dimension of the input MSA representation.
            pair_representation_embedding (int): The feature dimension of the input pair representation.
            number_heads (int): The total number of attention heads in each block.
            head_embedding_dimension (int): The dimensionality of each attention head in each block.
            channel_scaler (int): The scaling factor for hidden layers in each block.
            intermediate_embedding (int): The intermediate dimension for outer product mean.
            triangle_multiplication_embedding (int): The reduced hidden dimension for triangle updates.
            number_blocks (int): The total number of EvoformerBlocks to instantiate.
            single_representation_embedding (int): The feature dimension of the output single representation.
            device (torch.device): The computational device.
            dtype (torch.dtype): The numerical precision data type.
        """
        super().__init__()

        self.blocks = nn.ModuleList([
            EvoformerBlock(
                msa_embedding=msa_embedding,
                pair_representation_embedding=pair_representation_embedding,
                number_heads=number_heads,
                head_embedding_dimension=head_embedding_dimension,
                channel_scaler=channel_scaler,
                intermediate_embedding=intermediate_embedding,
                triangle_multiplication_embedding=triangle_multiplication_embedding,
                device=device, dtype=dtype
            ) for _ in range(number_blocks)
        ])

        self.msa_to_single_representation_embedder = nn.Linear(in_features=msa_embedding,
                                                               out_features=single_representation_embedding,
                                                               device=device,
                                                               dtype=dtype)

    def forward(self, msa_representation: torch.Tensor,
                pair_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the full Evoformer Stack.

        Args:
            msa_representation (torch.Tensor): The initial input MSA features.
                Shape: (..., number_sequences, number_residues, msa_embedding)
            pair_representation (torch.Tensor): The initial input pair features.
                Shape: (..., number_residues, number_residues, pair_representation_embedding)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The final updated MSA representation.
                  Shape: (..., number_sequences, number_residues, msa_embedding)
                - The final updated pair representation.
                  Shape: (..., number_residues, number_residues, pair_representation_embedding)
                - The single representation derived from the first row of the MSA.
                  Shape: (..., number_residues, single_representation_embedding)
        """
        for block in self.blocks:
            msa_representation, pair_representation = block(msa_representation, pair_representation)

        single_representation = self.msa_to_single_representation_embedder(msa_representation[..., 0, :, :])

        return msa_representation, pair_representation, single_representation
