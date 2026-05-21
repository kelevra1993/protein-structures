"""

"""

import torch
from torch import nn

from utilities.tensor_utilities import specialised_one_hot_encoder


class InputEmbedder(nn.Module):
    """
    """

    def __init__(self, input_sequence_feature_dimension: int,
                 input_msa_feature_dimension: int, input_extra_msa_feature_dimension: int,
                 msa_embedding: int, extra_msa_embeddding: int, pair_representation_embedding: int,
                 number_neighbouring_amino_acids: int, device: torch.device, dtype: torch.dtype):
        """

        """
        super().__init__()

        self.input_sequence_feature_dimension = input_sequence_feature_dimension
        self.input_msa_feature_dimension = input_msa_feature_dimension
        self.input_extra_msa_feature_dimension = input_extra_msa_feature_dimension
        self.msa_embedding = msa_embedding
        self.extra_msa_embedding = extra_msa_embeddding
        self.pair_representation_embedding = pair_representation_embedding
        self.number_neighbouring_amino_acids = number_neighbouring_amino_acids
        self.device = device
        self.dtype = dtype

        # Will Be Used For Relative Position Computation
        self.neighbouring_bins = torch.arange(start=-self.number_neighbouring_amino_acids,
                                              end=self.number_neighbouring_amino_acids + 1,
                                              step=1)

        # These Two Linear Layers Are Used For The Outer Sum Of Input Sequence Feature Creation
        self.input_sequence_pair_rep_embedder_i = nn.Linear(in_features=self.input_sequence_feature_dimension,
                                                            out_features=self.pair_representation_embedding,
                                                            dtype=self.dtype, device=self.device)
        self.input_sequence_pair_rep_embedder_j = nn.Linear(in_features=self.input_sequence_feature_dimension,
                                                            out_features=self.pair_representation_embedding,
                                                            dtype=self.dtype, device=self.device)

        # This Linear Layer Is Used For MSA Representation From Input Sequence Feature
        self.input_sequence_msa_rep_embedder = nn.Linear(in_features=self.input_sequence_feature_dimension,
                                                         out_features=self.msa_embedding,
                                                         dtype=self.dtype, device=self.device)

        # This Linear Layer Is Used For MSA Representation From Input MSA Feature
        self.input_msa_msa_rep_embedder = nn.Linear(in_features=self.input_msa_feature_dimension,
                                                    out_features=self.msa_embedding,
                                                    dtype=self.dtype, device=self.device)

        # This Linear Layer Is Used For Residue Index Embedding After It's Relative Position One Hot Encoding
        self.relative_position_embedder = nn.Linear(in_features=(2 * self.number_neighbouring_amino_acids) + 1,
                                                    out_features=self.pair_representation_embedding,
                                                    dtype=self.dtype, device=self.device)

    def compute_relative_positions(self, residue_index):
        """"""

        outer_difference = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        outer_difference = specialised_one_hot_encoder(input_tensor=outer_difference,
                                                       bin_tensor=self.neighbouring_bins)

        relative_positions = self.linear_relpos(outer_difference.to(self.dtype))

        return relative_positions

    def forward(self, input_sequence_feature, input_msa_feature, residue_index_feature):
        """"""

        input_embedded_i = self.input_sequence_pair_rep_embedder_i(input_sequence_feature)
        input_embedded_j = self.input_sequence_pair_rep_embedder_j(input_sequence_feature)

        input_outer_sum_embedding = input_embedded_i.unsqueeze(-2) + input_embedded_j.unsqueeze(-3)

        relative_position_embedding = self.relative_position_embedder(residue_index=residue_index_feature)
        pair_representation = input_outer_sum_embedding + relative_position_embedding

        input_feature_msa_embedded = self.input_sequence_msa_rep_embedder(input_sequence_feature).unsqueeze(-3)
        msa_feature_msa_embedded = self.input_msa_msa_rep_embedder(input_msa_feature)
        msa_representation = input_feature_msa_embedded + msa_feature_msa_embedded

        return msa_representation, pair_representation

# input_embedder = InputEmbedder()
