import torch

from typing import Optional, Tuple
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention



class TriangleMultiplication(nn.Module):

    def __init__(self, pair_representation_embedding, multiplication_type, embedding_dimension):

        super().__init__()

        self.multiplication_type = multiplication_type

        self.pair_representation_layer_normalizer = nn.LayerNorm(pair_representation_embedding)
        self.output_layer_normalizer = nn.LayerNorm(embedding_dimension)

        self.first_pair_representation_embedder = nn.Linear(pair_representation_embedding, embedding_dimension)
        self.first_gating_embedder = nn.Linear(pair_representation_embedding, embedding_dimension)

        self.second_pair_representation_embedder = nn.Linear(pair_representation_embedding, embedding_dimension)
        self.second_gating_embedder = nn.Linear(pair_representation_embedding, embedding_dimension)

        self.third_pair_representation_embedder = nn.Linear(pair_representation_embedding,
                                                            pair_representation_embedding)
        self.output_embedder = nn.Linear(embedding_dimension, pair_representation_embedding)

    def forward(self, pair_representation):

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
