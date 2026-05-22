import torch
from torch import nn

from utilities.tensor_utilities import specialised_one_hot_encoder, get_device


class InputEmbedder(nn.Module):
    """
    InputEmbedder module for AlphaFold II.

    This module is responsible for creating the initial MSA, pair, and extra MSA representations
    by embedding and combining various input features. It processes sequence features, MSA features,
    and residue indices to establish the foundation for subsequent architectural blocks.

    It performs:
    1.  Pair representation initialization via an outer sum of embedded sequence features.
    2.  Relative position embedding using residue indices.
    3.  MSA representation initialization by combining sequence and MSA features.
    4.  Extra MSA feature embedding.
    """

    def __init__(self, input_sequence_feature_dimension: int,
                 input_msa_feature_dimension: int, input_extra_msa_feature_dimension: int,
                 msa_embedding: int, extra_msa_embedding: int, pair_representation_embedding: int,
                 number_neighbouring_amino_acids: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the InputEmbedder with specified dimensions and settings.

        :param input_sequence_feature_dimension: Dimension of target sequence features (usually 21).
        :param input_msa_feature_dimension: Dimension of input MSA features (usually 49).
        :param input_extra_msa_feature_dimension: Dimension of extra MSA features (usually 25).
        :param msa_embedding: Hidden dimension for the MSA representation (msa_embedding_dimension).
        :param extra_msa_embedding: Hidden dimension for the extra MSA stack (extra_msa_embedding_dimension).
        :param pair_representation_embedding: Hidden dimension for the pair representation (pair_representation_dimension).
        :param number_neighbouring_amino_acids: Window size for relative position encoding.
        :param device: Hardware device for tensor operations.
        :param dtype: Data type for floating-point tensors.
        """
        super().__init__()

        self.input_sequence_feature_dimension = input_sequence_feature_dimension
        self.input_msa_feature_dimension = input_msa_feature_dimension
        self.input_extra_msa_feature_dimension = input_extra_msa_feature_dimension
        self.msa_embedding = msa_embedding
        self.extra_msa_embedding = extra_msa_embedding
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
        """
        Computes the relative position embedding between all pairs of residues.

        This method calculates the distance between every pair of residue indices,
        encodes these distances into a set of bins (one-hot), and embeds them into
        the pair representation space. This allows the model to be aware of the 
        linear distance along the sequence.

        :param residue_index: Tensor of shape (number_residues,) containing residue indices.
        :return: Relative position embedding of shape (number_residues, number_residues, pair_representation_dimension).
        """

        outer_difference = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        outer_difference = specialised_one_hot_encoder(input_tensor=outer_difference,
                                                       bin_tensor=self.neighbouring_bins)

        relative_positions = self.relative_position_embedder(outer_difference.to(dtype=self.dtype, device=self.device))

        return relative_positions

    def forward(self, input_sequence_feature, input_msa_feature, input_residue_index_feature, input_extra_msa_feature):
        """
        Executes the InputEmbedder forward pass to generate initial representations.

        This method transforms the raw features into the internal representations used by 
        the Evoformer and Extra MSA stacks. It combines target sequence information 
        with MSA and spatial (relative position) data.

        :param input_sequence_feature: Tensor of shape (number_residues, input_sequence_feature_dimension).
        :param input_msa_feature: Tensor of shape (number_clusters, number_residues, msa_feature_dimension).
        :param input_residue_index_feature: Tensor of shape (number_residues,).
        :param input_extra_msa_feature: Tensor of shape (number_extra_sequences, number_residues, input_extra_msa_feature_dimension).
        
        :return: A tuple containing:
            - msa_representation: (number_clusters, number_residues, msa_embedding_dimension).
            - pair_representation: (number_residues, number_residues, pair_representation_dimension).
            - extra_msa_representation: (number_extra_sequences, number_residues, extra_msa_embedding_dimension).
        """

        input_embedded_i = self.input_sequence_pair_rep_embedder_i(input_sequence_feature)
        input_embedded_j = self.input_sequence_pair_rep_embedder_j(input_sequence_feature)

        input_outer_sum_embedding = input_embedded_i.unsqueeze(-2) + input_embedded_j.unsqueeze(-3)

        relative_position_embedding = self.compute_relative_positions(residue_index=input_residue_index_feature)
        pair_representation = input_outer_sum_embedding + relative_position_embedding

        input_feature_msa_embedded = self.input_sequence_msa_rep_embedder(input_sequence_feature).unsqueeze(-3)
        msa_feature_msa_embedded = self.input_msa_msa_rep_embedder(input_msa_feature)
        msa_representation = input_feature_msa_embedded + msa_feature_msa_embedded

        extra_msa_representation = self.input_extra_msa_embedder(input_extra_msa_feature)

        return msa_representation, pair_representation, extra_msa_representation


if __name__ == "__main__":
    from pathlib import Path
    from utilities.tensor_utilities import print_tensor_shape
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

    computer_device = get_device()
    tensor_dtype = torch.float64

    # Initialize Input Embedder
    input_embedder = InputEmbedder(
        input_sequence_feature_dimension=extractor.input_sequence_feature.shape[-1],
        input_msa_feature_dimension=extractor.input_msa_feature.shape[-1],
        input_extra_msa_feature_dimension=extractor.input_extra_msa_feature.shape[-1],
        msa_embedding=256,
        extra_msa_embedding=64,
        pair_representation_embedding=128,
        number_neighbouring_amino_acids=32,
        device=computer_device,
        dtype=tensor_dtype
    )

    msa_rep, pair_rep, extra_msa_rep = input_embedder(
        input_sequence_feature=extractor.input_sequence_feature.to(device=computer_device, dtype=tensor_dtype),
        input_msa_feature=extractor.input_msa_feature.to(device=computer_device, dtype=tensor_dtype),
        input_residue_index_feature=extractor.input_residue_index_feature.to(device=computer_device, dtype=tensor_dtype),
        input_extra_msa_feature=extractor.input_extra_msa_feature.to(device=computer_device, dtype=tensor_dtype),
    )

    print_tensor_shape(name="MSA Representation",tensor=msa_rep)
    print_tensor_shape(name="Pair Representation",tensor=pair_rep)
    print_tensor_shape(name="Extra MSA Representation",tensor=extra_msa_rep)
