"""
This module contains the FeatureExtractor class, which is responsible for processing Multiple Sequence Alignment (MSA)
data from .a3m files into a set of features suitable for the AlphaFold II architecture. It handles sequence unique-ing,
MSA clustering, sequence masking, and the calculation of profiles and deletion features.

Naming Conventions for Tensor Shapes:
- number_residues: Length of the protein sequence.
- total_sequences: Total number of unique sequences in the MSA.
- number_clusters: Number of sequences selected for the MSA clusters (max maximum_cluster_sequences).
- number_extra_sequences: Number of sequences selected for the extra MSA stack (max maximum_extra_msa_sequences).
- number_canonical_amino_acids: 21 (20 canonical + 1 unknown 'X').
- number_gapped_amino_acids: 22 (number_canonical_amino_acids + 1 gap token '-').
- number_masked_amino_acids: 23 (number_gapped_amino_acids + 1 mask token).
- msa_feature_dimension: 49 (Sequence(23) + HasDeletion(1) + DeletionValue(1) + Profile(23) + DeletionMean(1)).
- extra_msa_feature_dimension: 25 (Sequence(23) + HasDeletion(1) + DeletionValue(1)).
"""

import torch

from torch import nn
from typing import Optional, List, Tuple

from utilities.constants import all_amino_acid_dictionary, gapped_amino_acid_dictionary
from utilities.tensor_utilities import print_tensor_shape, print_tensor_type, unsqueeze_tensor


class FeatureExtractor:
    """
    Handles the extraction and preprocessing of features from protein MSA data.

    This class processes raw .a3m files, performs MSA clustering, sequence masking,
    and computes evolutionary features such as profiles and deletion counts.
    """

    def __init__(self, file_path: str, maximum_cluster_sequences: int, maximum_extra_msa_sequences: int,
                 mask_probability: float, device: torch.device, dtype: torch.dtype,
                 seed: Optional[int] = None):
        """
        Initializes the FeatureExtractor and executes the feature extraction pipeline.

        :param file_path: Path to the .a3m file.
        :param maximum_cluster_sequences: Max number of sequences for the main MSA clusters.
        :param maximum_extra_msa_sequences: Max number of sequences for the extra MSA stack.
        :param mask_probability: Probability of masking residues in MSA clusters.
        :param device: Hardware device for tensor operations (e.g., torch.device('cuda')).
        :param dtype: Data type for floating-point tensors (e.g., torch.float32).
        :param seed: Random seed for sampling and masking.
        """

        self.file_path = file_path
        self.device = device
        self.dtype = dtype
        self.seed = seed

        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences

        # Fetch all sequences in the msa file List[str]
        self.unprocessed_sequences = self.load_a3m_file()

        # self.global_msa_sequence_tensor shape: (total_sequences, number_residues, number_gapped_amino_acids)
        # self.global_msa_deletion_count_tensor shape: (total_sequences, number_residues)
        self.global_msa_sequence_tensor, self.global_msa_deletion_count_tensor = self.compute_unique_sequences()

        # The target sequence (str).
        self.input_sequence = self.unprocessed_sequences[0]

        # shape: (number_residues, number_canonical_amino_acids)
        self.input_sequence_feature = self.one_hot_encode_amino_acid_types(
            sequence=self.input_sequence,
            include_gap_token=False)

        # shape: (number_residues,)
        self.residue_index_feature = torch.arange(len(self.input_sequence), device=self.device)

        self.number_residues = len(self.input_sequence)
        self.total_sequences = self.global_msa_sequence_tensor.shape[0]

        # shape: (1, number_residues, number_gapped_amino_acids)
        self.total_amino_acid_distribution = self.global_msa_sequence_tensor.mean(dim=0, keepdim=True)

        # self.input_msa_sequence_tensor shape: (number_clusters, number_residues, number_gapped_amino_acids)
        # self.input_extra_msa_sequence_tensor shape:
        # (number_extra_sequences, number_residues, number_gapped_amino_acids)
        # self.input_msa_deletion_count_tensor shape: (number_clusters, number_residues)
        # self.input_extra_msa_deletion_count_tensor shape: (number_extra_sequences, number_residues)
        (self.input_msa_sequence_tensor, self.input_extra_msa_sequence_tensor,
         self.input_msa_deletion_count_tensor, self.input_extra_msa_deletion_count_tensor) = (
            self.select_cluster_centers(seed=self.seed))

        self.number_clusters = self.input_msa_sequence_tensor.shape[0]
        self.number_extra_sequences = self.input_extra_msa_sequence_tensor.shape[0]

        self.mask_probability = mask_probability
        # Modifies self.input_msa_sequence_tensor to shape:
        # (number_clusters, number_residues, number_masked_amino_acids)
        self.mask_cluster_centers(seed=self.seed)

        # self.assignments_tensor shape: (number_extra_sequences,)
        # self.assignments_count_tensor shape: (number_clusters,)
        self.assignments_tensor, self.assignments_count_tensor = self.assign_cluster()

        # self.cluster_deletion_contributions shape: (number_clusters, number_residues)
        # self.cluster_profile shape: (number_clusters, number_residues, number_masked_amino_acids)
        self.cluster_deletion_contributions, self.cluster_profile = self.summarize_extra_msa_feature_to_kept_msa()

        # Crops extra sequences to maximum_extra_msa_sequences.
        self.crop_extra_msa_features(seed=self.seed)

        # self.input_cluster_deletion_value shape: (number_clusters, number_residues, 1)
        # self.input_cluster_has_deletion shape: (number_clusters, number_residues, 1)
        # self.input_extra_msa_deletion_value shape: (number_extra_sequences, number_residues, 1)
        # self.input_extra_msa_has_deletion shape: (number_extra_sequences, number_residues, 1)
        (self.input_cluster_deletion_value, self.input_cluster_has_deletion, self.input_extra_msa_deletion_value,
         self.input_extra_msa_has_deletion) = self.get_deletion_input_features()

        # shape: (number_clusters, number_residues, msa_feature_dimension)
        self.input_msa_feature = torch.cat(tensors=[
            self.input_msa_sequence_tensor,
            self.input_cluster_has_deletion,
            self.input_cluster_deletion_value,
            self.cluster_profile,
            self.cluster_deletion_contributions.unsqueeze(-1)], dim=-1)

        # shape: (number_extra_sequences, number_residues, extra_msa_feature_dimension)
        self.input_extra_msa_feature = torch.cat(tensors=[
            # Adding masked token column to mimic masked input structure
            torch.nn.functional.pad(self.input_extra_msa_sequence_tensor, (0, 1), value=0),
            self.input_extra_msa_has_deletion,
            self.input_extra_msa_deletion_value], dim=-1)

        print_tensor_shape(self.input_msa_feature)
        print_tensor_type(self.input_msa_feature)
        print_tensor_shape(self.input_extra_msa_feature)
        print_tensor_type(self.input_extra_msa_feature)

    def load_a3m_file(self) -> List[str]:
        """
        Parses the .a3m file and extracts protein sequences.

        Lines starting with '>' are treated as headers and ignored. The following line
        is extracted as a sequence.

        :return: A list of sequence strings extracted from the file.
        """

        with open(self.file_path, "r") as msa_data:
            content = msa_data.readlines()
            sequences = [content[index + 1].strip() for index, line in enumerate(content) if line[0] == ">"]

        return sequences

    def one_hot_encode_amino_acid_types(self, sequence: str, include_gap_token: bool = False) -> torch.Tensor:
        """
        Converts a sequence string into a one-hot encoded tensor.

        The dictionary used depends on whether gap tokens are included. This distinction
        is necessary because the target input sequence feature typically excludes gaps,
        while MSA sequences include them.

        - If include_gap_token is False:
          Shape: (number_residues, number_canonical_amino_acids)
        - If include_gap_token is True:
          Shape: (number_residues, number_gapped_amino_acids)

        :param sequence: The amino acid sequence string to encode.
        :param include_gap_token: Whether to include the gap token '-' in the encoding categories.
        :return: A one-hot encoded tensor of the sequence.
        """

        amino_acid_dictionary = all_amino_acid_dictionary if not include_gap_token else gapped_amino_acid_dictionary

        sequence_indices = torch.tensor([amino_acid_dictionary[amino_acid_index] for amino_acid_index in sequence],
                                        device=self.device)

        encoding = torch.nn.functional.one_hot(sequence_indices, num_classes=len(amino_acid_dictionary))

        return encoding.to(self.dtype)

    def compute_unique_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes unprocessed sequences to remove duplicates and track insertions.

        Insertions (represented by lowercase letters in .a3m) are removed to maintain
        a consistent protein length across the MSA. The number of consecutive insertions
        to the left of each residue is tracked in a deletion count matrix.

        - unique_sequences_tensor shape: (total_sequences, number_residues, number_gapped_amino_acids)
        - deletion_count_matrix shape: (total_sequences, number_residues)

        :return: A tuple containing the unique gapped sequence tensor and the deletion count tensor.
        """
        deletion_count_matrix = []
        unique_sequences = []

        for sequence in self.unprocessed_sequences:

            processed_sequence = ""
            sequence_deletion_list = []
            temporary_deletion_count = 0

            for amino_acid in sequence:
                if amino_acid.islower():
                    temporary_deletion_count += 1
                    continue

                processed_sequence += amino_acid
                sequence_deletion_list.append(temporary_deletion_count)
                temporary_deletion_count = 0

            if processed_sequence not in unique_sequences:
                unique_sequences.append(processed_sequence)
                deletion_count_matrix.append(sequence_deletion_list)

        # Turn deletion count matrix into a tensor
        deletion_count_matrix = torch.tensor(deletion_count_matrix, dtype=self.dtype, device=self.device)

        unique_sequences_matrix = [self.one_hot_encode_amino_acid_types(sequence, include_gap_token=True) for sequence
                                   in unique_sequences]
        unique_sequences_tensor = torch.stack(unique_sequences_matrix, dim=0).to(device=self.device, dtype=self.dtype)

        return unique_sequences_tensor, deletion_count_matrix

    def select_cluster_centers(self, seed: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shuffles unique sequences and partitions them into MSA clusters and extra sequences.

        The first sequence (the target) is always preserved at the first position.
        The remaining sequences are shuffled and split based on the maximum cluster
        and extra sequence limits.

        - input_msa_sequence_tensor shape: (number_clusters, number_residues, number_gapped_amino_acids)
        - extra_msa_sequence_tensor shape: (number_extra_sequences, number_residues, number_gapped_amino_acids)
        - input_msa_deletion_count_tensor shape: (number_clusters, number_residues)
        - extra_msa_deletion_count_tensor shape: (number_extra_sequences, number_residues)

        :param seed: Optional seed for the random permutation to ensure reproducibility.
        :return: A tuple containing the partitioned sequence and deletion count tensors.
        """

        max_msa_clusters = min(self.maximum_cluster_sequences, self.total_sequences)

        gen = None
        if seed is not None:
            gen = torch.Generator(self.global_msa_sequence_tensor.device)
            gen.manual_seed(seed)

        # Create Shuffling indices
        shuffled_indices = torch.cat((torch.tensor([0]),
                                      torch.randperm(n=self.total_sequences - 1, generator=gen) + 1), dim=0)

        # Separate MSA Sequences In Two (Main and Extra MSA Sequence)
        input_msa_sequence_tensor = self.global_msa_sequence_tensor[shuffled_indices][:max_msa_clusters]
        extra_msa_sequence_tensor = self.global_msa_sequence_tensor[shuffled_indices][max_msa_clusters:]

        input_msa_deletion_count_tensor = self.global_msa_deletion_count_tensor[shuffled_indices][:max_msa_clusters]
        extra_msa_deletion_count_tensor = self.global_msa_deletion_count_tensor[shuffled_indices][max_msa_clusters:]

        return (input_msa_sequence_tensor, extra_msa_sequence_tensor,
                input_msa_deletion_count_tensor, extra_msa_deletion_count_tensor)

    def mask_cluster_centers(self, seed: Optional[int] = None) -> None:
        """
        Applies a masking strategy to the MSA cluster centers.

        A fraction of residues (defined by mask_probability) are selected for modification.
        Selected residues are either replaced with a special mask token, a random amino acid,
        the actual amino acid from the distribution, or left unchanged. This process
        increases the dimensionality of the sequence tensor to include the mask token.

        Modifies self.input_msa_sequence_tensor in-place.
        Resulting shape: (number_clusters, number_residues, number_masked_amino_acids)

        :param seed: Optional seed for the masking process to ensure reproducibility.
        :return: None
        """
        number_amino_acid_categories = 23  # 20 Amino Acids, Unknown AA, Gap, masked_msa_token

        odds = {
            'uniform': 0.1,
            'amino_acid_distribution': 0.1,
            'no_action': 0.1,
            'add_mask': 0.7,
        }

        gen = None
        if seed is not None:
            gen = torch.Generator(self.input_msa_sequence_tensor.device)
            gen.manual_seed(seed)
            torch.manual_seed(seed)

        random_mask = torch.rand((self.number_clusters, self.number_residues), generator=gen)
        indices_to_change = random_mask < self.mask_probability

        # Uniform Replacement (number_sequences, number_residues, 22)
        uniform_replacement = torch.tensor([1 / 20] * 20 + [0, 0]) * odds['uniform']

        # From Amino Acid Distribution
        from_distribution_replacement = self.total_amino_acid_distribution * odds['amino_acid_distribution']

        # No Replacement
        no_replacement = self.input_msa_sequence_tensor * odds['no_action']

        # Add Mask
        masked_out = torch.ones((self.number_clusters, self.number_residues, 1)) * odds['add_mask']

        # Summing initial categories up
        # Note : This Will Broadcast Until Shape Of (number_cluster, number_residues, 22)
        categories_with_mask_token = uniform_replacement.reshape((1, 1, 22)) + from_distribution_replacement
        categories_with_mask_token = categories_with_mask_token + no_replacement

        categories_with_mask_token = torch.cat(tensors=(categories_with_mask_token, masked_out), dim=-1).to(
            device=self.device, dtype=self.dtype)

        # Reshaped To (number_cluster * number_residues, 23)
        categories_with_mask_token = categories_with_mask_token.reshape(-1, number_amino_acid_categories)

        # .sample() gives out a two dimension matrix (number_cluster * number_residues, 1) "Selected Indices"
        replacement_indices = torch.distributions.Categorical(categories_with_mask_token).sample()
        replacement_tensor = nn.functional.one_hot(replacement_indices, num_classes=number_amino_acid_categories)
        replacement_tensor = replacement_tensor.reshape(self.number_clusters, self.number_residues,
                                                        number_amino_acid_categories)
        replacement_tensor = replacement_tensor.to(device=self.device, dtype=self.dtype)

        # Modifies The Input MSA Sequence Tensor
        self.input_msa_sequence_tensor = nn.functional.pad(self.input_msa_sequence_tensor, (0, 1), value=0)
        self.input_msa_sequence_tensor[indices_to_change] = replacement_tensor[indices_to_change].to(dtype=self.dtype)

    def assign_cluster(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assigns each extra MSA sequence to its nearest cluster center.

        The assignment is based on the agreement score (dot product) between the extra
        sequence and the cluster center sequences, excluding gaps and masked tokens.

        - assignments_tensor shape: (number_extra_sequences,)
        - assignments_count_tensor shape: (number_clusters,)

        :return: A tuple containing the assignment indices and the count of sequences per cluster.
        """

        # Removing Masked Tokens And Gaps
        sliced_input_msa_sequence = self.input_msa_sequence_tensor[..., :21]
        sliced_extra_msa_sequence = self.input_extra_msa_sequence_tensor[..., : 21]

        agreement_score = torch.einsum("...abc, ...dbc -> ad", sliced_extra_msa_sequence, sliced_input_msa_sequence)
        assignments_tensor = torch.argmax(agreement_score, dim=-1)
        assignments_count_tensor = torch.bincount(assignments_tensor, minlength=self.number_clusters)

        return assignments_tensor.to(device=self.device), assignments_count_tensor.to(device=self.device,
                                                                                      dtype=self.dtype)

    def cluster_average(self, feature: torch.Tensor, extra_feature: torch.Tensor) -> torch.Tensor:
        """
        Computes the average feature value for each cluster using assigned extra MSA data.

        This method aggregates features from extra MSA sequences into their assigned
        cluster centers and calculates the mean, accounting for the original cluster
        center sequence itself.

        - feature shape: (number_clusters, number_residues, ...)
        - extra_feature shape: (number_extra_sequences, number_residues, ...)
        - Resulting shape: (number_clusters, number_residues, ...)

        :param feature: The initial feature tensor for the cluster centers.
        :param extra_feature: The feature tensor for the extra MSA sequences.
        :return: The averaged feature tensor for the clusters.
        """

        missing_dimensions = feature.ndim - 1
        assignments_tensor = unsqueeze_tensor(self.assignments_tensor, direction="right", number=missing_dimensions)
        assignments_counts = unsqueeze_tensor(self.assignments_count_tensor, direction="right",
                                              number=missing_dimensions)

        # Broadcast To Match Extra MSA Feature Tensors
        assignments_tensor = torch.broadcast_to(assignments_tensor, size=extra_feature.shape)

        # Broadcast To Match MSA Feature Tensors (but inherently not necessary)
        cluster_assignment_count = torch.broadcast_to(assignments_counts, size=feature.shape)

        accumulated_features = torch.scatter_add(input=feature, src=extra_feature, dim=0, index=assignments_tensor)
        cluster_average = accumulated_features * 1 / (cluster_assignment_count + 1)

        return cluster_average

    def summarize_extra_msa_feature_to_kept_msa(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates extra MSA information into profiles and deletion contributions for clusters.

        This method uses cluster averaging to summarize the information from the
        extra MSA stack into the main MSA clusters.

        - cluster_deletion_contributions shape: (number_clusters, number_residues)
        - cluster_profile shape: (number_clusters, number_residues, number_masked_amino_acids)

        :return: A tuple of (cluster deletion contributions, cluster profile).
        """

        # Compute Contribution Of Extra Sequences To MSA Sequence (for deletion counts)
        cluster_deletion_contributions = self.cluster_average(
            feature=self.input_msa_deletion_count_tensor,
            extra_feature=self.input_extra_msa_deletion_count_tensor)

        cluster_deletion_contributions = 2 / torch.pi * torch.arctan(cluster_deletion_contributions / 3)

        # Compute Contribution Of Extra Sequences To MSA Sequence
        cluster_profile = self.cluster_average(
            feature=self.input_msa_sequence_tensor,
            extra_feature=nn.functional.pad(self.input_extra_msa_sequence_tensor, (0, 1), value=0))

        return cluster_deletion_contributions, cluster_profile

    def crop_extra_msa_features(self, seed: Optional[int] = None) -> None:
        """
        Limits the number of extra MSA features to the specified maximum.

        This reduction is performed by randomly sampling from the available extra
        sequences.

        Modifies self.input_extra_msa_sequence_tensor and self.input_extra_msa_deletion_count_tensor.

        :param seed: Optional seed for the random permutation to ensure reproducibility.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(self.input_extra_msa_sequence_tensor.device)
            gen.manual_seed(seed)

        max_extra_msa_count = min(self.maximum_extra_msa_sequences, self.number_extra_sequences)

        shuffled_indices = torch.randperm(n=self.number_extra_sequences, generator=gen)
        sliced_shuffled_indices = shuffled_indices[:max_extra_msa_count]

        self.input_extra_msa_sequence_tensor = self.input_extra_msa_sequence_tensor[sliced_shuffled_indices]
        self.input_extra_msa_deletion_count_tensor = self.input_extra_msa_deletion_count_tensor[sliced_shuffled_indices]

    def get_deletion_input_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transforms raw deletion counts into final model input features.

        Raw deletion counts are transformed using an arctan function to a [0, 1] range.
        Additionally, a binary feature is created to indicate the presence of a deletion.

        - input_cluster_deletion_value shape: (number_clusters, number_residues, 1)
        - input_cluster_has_deletion shape: (number_clusters, number_residues, 1)
        - extra_msa_deletion_value shape: (number_extra_sequences, number_residues, 1)
        - extra_msa_has_deletion shape: (number_extra_sequences, number_residues, 1)

        :return: A tuple containing deletion values and indicators for clusters and extra sequences.
        """

        # Transformed Range For Deletion Counts
        input_cluster_deletion_value = 2 / torch.pi * torch.arctan(self.input_msa_deletion_count_tensor / 3)
        input_cluster_deletion_value = input_cluster_deletion_value.unsqueeze(dim=-1)

        # Get Deletion Presence To The Left Of The Residues
        input_cluster_has_deletion = (input_cluster_deletion_value > 0)

        extra_msa_deletion_value = 2 / torch.pi * torch.arctan(self.input_extra_msa_deletion_count_tensor / 3)
        extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(dim=-1)

        extra_msa_has_deletion = (extra_msa_deletion_value > 0)

        return input_cluster_deletion_value, input_cluster_has_deletion, extra_msa_deletion_value, extra_msa_has_deletion
