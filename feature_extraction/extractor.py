"""
Todo add description
# TODO Add specification of shapes :
(number_sequences, number_residues,22)
(number_sequences, number_residues,22)
"""
import torch

from torch import nn
from typing import Optional


# Todo make sure that every time we create tensor we have set the device and the dtype
from utilities.constants import all_amino_acid_dictionary, gapped_amino_acid_dictionary
from utilities.tensor_utilities import get_device, print_shape, unsqueeze_tensor


class FeatureExtractor:
    """
    Todo add class description
    Todo add device type
    Todo add dtype type
    """

    def __init__(self, file_path: str, maximum_cluster_sequences: int, maximum_extra_msa_sequences: int,
                 mask_probability: float, device, dtype,
                 seed: Optional[int] = None):
        """
        # Add documentation
        :param file_path:
        """

        self.file_path = file_path
        self.device = device
        self.dtype = dtype
        self.seed = seed

        # Maximum Number Of Sequences Per MSA Cluster (That Goes Into The MSA Embedder)
        # Maximum Number Of Sequences In Extra MSA (That Goes Into The MSA Stack)
        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences

        # Fetch all sequences in the msa file
        self.unprocessed_sequences = self.load_a3m_file()

        # Fetch MSA Sequence Tensor : Removing Duplications After 'insertion' removal
        # Fetch Deletion Count Keeping Track Of Consecutive Insertion On The Left
        self.msa_sequence_tensor, self.msa_deletion_count_tensor = self.compute_unique_sequences()

        # Get Input Sequence Tensor (Un-modified) -> shape : (number_residues, 21)
        self.input_sequence = self.msa_sequence_tensor[0]
        self.input_sequence_feature = self.one_hot_encode_amino_acid_types(
            sequence=self.input_sequence,
            include_gap_token=False)

        # Residue Index Tensor That Will Be Used For Embedding -> shape (number_residues,)
        self.residue_index_feature = torch.arange(len(self.input_sequence))

        # Number Of Residues And Total Sequences
        self.number_residues = len(self.unprocessed_sequences[0])
        self.total_sequences = self.msa_sequence_tensor.shape[0]

        # Amino Acid Distribution Used For Masking
        self.total_amino_acid_distribution = self.msa_sequence_tensor.mean(dim=0, keepdim=True)

        # Separate Between Main MSA Used For MSA Embedder And Extra MSA Used For Extra MSA Embedder
        (self.input_msa_sequence_tensor, self.extra_msa_sequence_tensor,
         self.input_msa_deletion_count_tensor, self.extra_msa_deletion_count_tensor) = self.select_cluster_centers(
            seed=self.seed)

        # Number Of Element In The Input Cluster And Extra Sequences
        self.number_clusters = self.input_msa_sequence_tensor.shape[0]
        self.number_extra_sequences = self.extra_msa_sequence_tensor.shape[0]

        # Masking Probability For More Robust Training (Affects self.input_msa_sequence_tensor)
        self.mask_probability = mask_probability
        self.mask_cluster_centers(seed=self.seed)

        # Cluster Assignments For Extra MSA Sequences And Associated Assignment Counts On Input MSA Sequence
        self.assignments_tensor, self.assignments_count_tensor = self.assign_cluster()

        # Compute Contributions Of Extra MSA Data To MSA Data, While Averaging It All
        self.cluster_deletion_contributions, self.cluster_profile = self.summarize_extra_msa_feature_to_kept_msa()

        # Limit Number Of Extra MSA Features
        # Affects self.extra_msa_sequence_tensor And self.extra_msa_deletion_count_tensor
        self.crop_extra_msa_features(seed=self.seed)

        # Get Deletion Values That Will Ultimately Be In The MSA Input And Extra MSA Input
        (self.input_cluster_deletion_value, self.input_cluster_has_deletion, self.extra_msa_deletion_value,
         self.extra_msa_has_deletion) = self.get_deletion_input_features()

        # Create Input MSA Feature
        self.input_msa_feature = torch.cat(tensors=[
            self.input_msa_sequence_tensor,
            self.input_cluster_has_deletion,
            self.input_cluster_deletion_value,
            self.cluster_profile,
            self.cluster_deletion_contributions.unsqueeze(-1)], dim=-1)

        # Create Input Extra MSA Feature
        self.input_extra_msa_feature = torch.cat(tensors=[
            # Adding 23 column to mimic masked input
            torch.nn.functional.pad(self.extra_msa_sequence_tensor, (0, 1), value=0),
            self.extra_msa_has_deletion,
            self.extra_msa_deletion_value], dim=-1)

        print_shape(self.input_msa_feature)
        print_shape(self.input_extra_msa_feature)

    # Todo Implement loading of a3m file.
    def load_a3m_file(self):
        """
        Todo to be documented
        :return:
        """

        with open(self.file_path, "r") as msa_data:
            content = msa_data.readlines()
            sequences = [content[index + 1].strip() for index, line in enumerate(content) if line[0] == ">"]

        return sequences

    # Todo Implement Onehot encoder with gap option.
    # Add information about the fact that we have the include_gap_token boolean because of input_sequence_feature does not include this gap
    def one_hot_encode_amino_acid_types(self, sequence, include_gap_token=False):
        """
        todo add documentation
        Add shapes in documentation whenever possible
        :param sequence:
        :param include_gap_token:
        :return:
        """

        amino_acid_dictionary = all_amino_acid_dictionary if not include_gap_token else gapped_amino_acid_dictionary

        sequence_indices = torch.tensor([amino_acid_dictionary[amino_acid_index] for amino_acid_index in sequence],
                                        device=self.device)

        encoding = torch.nn.functional.one_hot(sequence_indices, num_classes=len(amino_acid_dictionary))

        return encoding.to(self.dtype)

    # Todo Compute Unique Sequences (outputs unique sequences + deletion matrix )
    # - properly document deletion count
    def compute_unique_sequences(self):
        """
        Todo to be documented
        Add shapes in documentation
        // removes insertions in homolog sequences (considered as deletions) and keeps a track of them
        // counting number of consecutive deletions on the left of the residue
        :return:
        """
        # todo rename temporary count to temporary deletion counter
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

    # Todo Select Cluster Sequences
    # Todo Review Seed, Only used for testing
    # - Yields four outputs msa_cluster, msa_deletion_count and their 'extra_' counterparts
    def select_cluster_centers(self, seed: int | None = None):
        """"""

        max_msa_clusters = min(self.maximum_cluster_sequences, self.total_sequences)

        # TODO Try to understand this seed
        gen = None
        if seed is not None:
            gen = torch.Generator(self.msa_sequence_tensor.device)
            gen.manual_seed(seed)

        # Create Shuffling indices
        shuffled_indices = torch.cat((torch.tensor([0]),
                                      torch.randperm(n=self.total_sequences - 1, generator=gen) + 1), dim=0)

        # Separate MSA Sequences In Two (Main and Extra MSA Sequence)
        input_msa_sequence_tensor = self.msa_sequence_tensor[shuffled_indices][:max_msa_clusters]
        extra_msa_sequence_tensor = self.msa_sequence_tensor[shuffled_indices][max_msa_clusters:]

        input_msa_deletion_count_tensor = self.msa_deletion_count_tensor[shuffled_indices][:max_msa_clusters]
        extra_msa_deletion_count_tensor = self.msa_deletion_count_tensor[shuffled_indices][max_msa_clusters:]

        return (input_msa_sequence_tensor, extra_msa_sequence_tensor,
                input_msa_deletion_count_tensor, extra_msa_deletion_count_tensor)

    # Todo Apply Masking
    def mask_cluster_centers(self, seed: Optional[int] = None):
        """
        todo add documentation
        :param seed:
        :return:
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

        # todo reconsider this flatten to the proposed reshaped that is actually more readible
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

    def assign_cluster(self):
        """
        """

        # TODO Rename this to sliced : Better for understanding
        # Removing Masked Tokens And Gaps
        sliced_input_msa_sequence = self.input_msa_sequence_tensor[..., :21]
        sliced_extra_msa_sequence = self.extra_msa_sequence_tensor[..., : 21]

        agreement_score = torch.einsum("...abc, ...dbc -> ad", sliced_extra_msa_sequence, sliced_input_msa_sequence)
        assignments_tensor = torch.argmax(agreement_score, dim=-1)
        assignments_count_tensor = torch.bincount(assignments_tensor, minlength=self.number_clusters)

        return assignments_tensor.to(device=self.device), assignments_count_tensor.to(device=self.device,
                                                                                      dtype=self.dtype)

    # TODO Function For Cluster Averaging using extra msa to assigned cluster centers
    # - Must be properly explained
    # todo : Note only applied for deletion and profile ????
    def cluster_average(self, feature: torch.Tensor, extra_feature: torch.Tensor):
        """"""

        # todo Add the shapes of the input tensors in the docstring
        missing_dimensions = feature.ndim - 1
        assignments_tensor = unsqueeze_tensor(self.assignments_tensor, direction="right", number=missing_dimensions)
        assignments_counts = unsqueeze_tensor(self.assignments_count_tensor, direction="right",
                                              number=missing_dimensions)

        # Broadcast To Match Extra MSA Feature Tensors
        assignments_tensor = torch.broadcast_to(assignments_tensor, size=extra_feature.shape)

        # Broadcast To Match MSA Feature Tensors (but inherently not necessary)
        # print_shape(assignments_counts)
        cluster_assignment_count = torch.broadcast_to(assignments_counts, size=feature.shape)

        accumulated_features = torch.scatter_add(input=feature, src=extra_feature, dim=0, index=assignments_tensor)
        cluster_average = accumulated_features * 1 / (cluster_assignment_count + 1)

        return cluster_average

    # TODO Apply Cluster Averaging for deletion and profiling
    # - Must be properly explained
    def summarize_extra_msa_feature_to_kept_msa(self):
        """"""

        # Compute Contribution Of Extra Sequences To MSA Sequence (for deletion counts)
        cluster_deletion_contributions = self.cluster_average(
            feature=self.input_msa_deletion_count_tensor,
            extra_feature=self.extra_msa_deletion_count_tensor)

        cluster_deletion_contributions = 2 / torch.pi * torch.arctan(cluster_deletion_contributions / 3)

        # Compute Contribution Of Extra Sequences To MSA Sequence
        cluster_profile = self.cluster_average(
            feature=self.input_msa_sequence_tensor,
            extra_feature=nn.functional.pad(self.extra_msa_sequence_tensor, (0, 1), value=0))

        return cluster_deletion_contributions, cluster_profile

    # TODO Crop extra msa count
    def crop_extra_msa_features(self, seed=None):
        """"""
        gen = None
        if seed is not None:
            gen = torch.Generator(self.extra_msa_sequence_tensor.device)
            gen.manual_seed(seed)

        max_extra_msa_count = min(self.maximum_extra_msa_sequences, self.number_extra_sequences)

        shuffled_indices = torch.randperm(n=self.number_extra_sequences, generator=gen)
        sliced_shuffled_indices = shuffled_indices[:max_extra_msa_count]

        self.extra_msa_sequence_tensor = self.extra_msa_sequence_tensor[sliced_shuffled_indices]
        self.extra_msa_deletion_count_tensor = self.extra_msa_deletion_count_tensor[sliced_shuffled_indices]

    # TODO Create Full MSA Feature
    def get_deletion_input_features(self):
        """"""

        # Transformed Range For Deletion Counts
        input_cluster_deletion_value = 2 / torch.pi * torch.arctan(self.input_msa_deletion_count_tensor / 3)
        input_cluster_deletion_value = input_cluster_deletion_value.unsqueeze(dim=-1)

        # Get Deletion Presence To The Left Of The Residues
        input_cluster_has_deletion = (input_cluster_deletion_value > 0)

        extra_msa_deletion_value = 2 / torch.pi * torch.arctan(self.extra_msa_deletion_count_tensor / 3)
        extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(dim=-1)

        extra_msa_has_deletion = (extra_msa_deletion_value > 0)

        return input_cluster_deletion_value, input_cluster_has_deletion, extra_msa_deletion_value, extra_msa_has_deletion


features = FeatureExtractor(
    file_path="multiple_sequence_alignement.a3m",
    maximum_cluster_sequences=512,
    maximum_extra_msa_sequences=5120,
    mask_probability=0.15,
    device=torch.device("cpu"),
    dtype=torch.float64,
    seed=0)
