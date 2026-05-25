import torch
import math
from torch import nn

from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector




class InvariantPointAttention(nn.Module):

    def __init__(self, single_representation_embedding: int, pair_representation_embedding: int,
                 number_query_points: int = 4, number_value_points: int = 8, number_heads: int = 12,
                 head_embedding_dimension: int = 16, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_heads = number_heads
        self.head_embedding_dimension = head_embedding_dimension
        self.device = device
        self.dtype = dtype

        # todo : here alpha fold implementation did not use bias but we will use Bias
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

