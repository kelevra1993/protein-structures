import torch
import numpy as np

from typing import Optional, Tuple
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention




class MSARowAttentionWithPairBias(nn.Module):
    """"""

    def __init__(self, msa_embedding: int, pair_representation_embedding: int, head_embedding_dimension: int,
                 number_heads: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=msa_embedding)
        self.pair_representation_layer_normalizer = nn.LayerNorm(normalized_shape=pair_representation_embedding)

        # todo recheck but this layer does not use bias
        self.pair_representation_embedder = nn.Linear(in_features=pair_representation_embedding,
                                                      out_features=number_heads, bias=False)

        self.multi_head_attention = MultiHeadAttention(input_dimension=msa_embedding,
                                                       head_embedding_dimension=head_embedding_dimension,
                                                       number_heads=number_heads,
                                                       attention_dimension=-2,
                                                       use_gating=True,
                                                       use_global_attention=False,
                                                       use_embedding_bias=False,
                                                       device=device,
                                                       dtype=dtype)

    def forward(self, msa_representation: torch.Tensor, pair_representation: torch.Tensor) -> torch.Tensor:
        normalized_msa_representation = self.msa_representation_layer_normalizer(msa_representation)

        # Creation of bias
        normalized_pair_representaion = self.pair_representation_layer_normalizer(pair_representation)
        bias_tensor = self.pair_representation_embedder(normalized_pair_representaion).movedim(source=-1, destination=-3)

        output_tensor = self.multi_head_attention.forward(input_tensor=normalized_msa_representation,
                                                          bias_tensor=bias_tensor)

        return output_tensor
