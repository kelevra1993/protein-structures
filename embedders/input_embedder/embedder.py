"""
c_m -> msa_embedding
c_z -> pair_representation_embedding
tf_dim -> input feature embedding or input_sequence_feature_dimension
(basically normally just the number of canonical amino acids ? Just to be restated)
msa_feat_dim -> input_msa_feature_dimension
vbins -> Number of relative amino acids to consider on the left and on the right of our amino acid :
 number_neighbooring_amino_acids
f_e -> input_extra_msa_feature_dimension (normally 25)
c_e -> extra_msa_embeddding
"""
import os

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
                                              step=1,
                                              device=self.device)

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

        # This Linear Layer Is For The Extra MSA Stack Embeddor
        self.input_extra_msa_embedder = nn.Linear(in_features=self.input_extra_msa_feature_dimension,
                                                  out_features=self.extra_msa_embedding,
                                                  dtype=self.dtype, device=self.device)

    def compute_relative_positions(self, residue_index):
        """"""

        outer_difference = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        outer_difference = specialised_one_hot_encoder(input_tensor=outer_difference,
                                                       bin_tensor=self.neighbouring_bins)

        relative_positions = self.relative_position_embedder(outer_difference.to(dtype=self.dtype, device=self.device))

        return relative_positions

    def forward(self, input_sequence_feature, input_msa_feature, residue_index_feature,extra_msa_feature):
        """"""

        input_embedded_i = self.input_sequence_pair_rep_embedder_i(input_sequence_feature)
        input_embedded_j = self.input_sequence_pair_rep_embedder_j(input_sequence_feature)

        input_outer_sum_embedding = input_embedded_i.unsqueeze(-2) + input_embedded_j.unsqueeze(-3)

        relative_position_embedding = self.compute_relative_positions(residue_index=residue_index_feature)
        pair_representation = input_outer_sum_embedding + relative_position_embedding

        input_feature_msa_embedded = self.input_sequence_msa_rep_embedder(input_sequence_feature).unsqueeze(-3)
        msa_feature_msa_embedded = self.input_msa_msa_rep_embedder(input_msa_feature)
        msa_representation = input_feature_msa_embedded + msa_feature_msa_embedded

        extra_msa_representation = self.input_extra_msa_embedder(extra_msa_feature)

        return msa_representation, pair_representation, extra_msa_representation


from pathlib import Path
from feature_extraction.extractor import FeatureExtractor

# Robust path to the test file
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
msa_file_path = project_root / "tests" / "feature_extraction" / "multiple_sequence_alignement.a3m"

if not msa_file_path.exists():
    # Fallback for different execution contexts
    msa_file_path = project_root / "test" / "multiple_sequence_alignement.a3m"

# Initialize the extractor with fixed parameters and seed for determinism
extractor = FeatureExtractor(
    file_path=str(msa_file_path),
    maximum_cluster_sequences=512,
    maximum_extra_msa_sequences=5120,
    mask_probability=0.15,
    device=torch.device("cpu"),
    dtype=torch.float32,
    seed=0
)

input_embedder = InputEmbedder(
    input_sequence_feature_dimension=extractor.input_sequence_feature.shape[-1],
    input_msa_feature_dimension=extractor.input_msa_feature.shape[-1],
    input_extra_msa_feature_dimension=extractor.input_extra_msa_feature.shape[-1],
    msa_embedding=256,
    extra_msa_embeddding=64,
    pair_representation_embedding=128,
    number_neighbouring_amino_acids=32,
    device=torch.device("mps"),
    dtype=extractor.dtype
)

msa_rep, pair_rep,extra_msa_rep = input_embedder(
    input_sequence_feature=extractor.input_sequence_feature.to(torch.device("mps")),
    input_msa_feature=extractor.input_msa_feature.to(torch.device("mps")),
    residue_index_feature=extractor.residue_index_feature.to(torch.device("mps")),
    extra_msa_feature=extractor.input_extra_msa_feature.to(torch.device("mps"))
)

print(f"MSA representation shape: {msa_rep.shape}")
print(f"Pair representation shape: {pair_rep.shape}")
print(f"Extra MSA representation shape: {extra_msa_rep.shape}")
