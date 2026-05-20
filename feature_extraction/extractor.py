"""
Todo add description
# TODO Add specification of shapes :
(number_sequences, number_residues,22)
(number_sequences, number_residues,22)
"""
import torch

from torch import nn
from typing import Optional

# Todo Use math for the arctan ?
import math

# Todo make sure that every time we create tensor we have set the device and the dtype

from utilities.constants import all_amino_acid_dictionary, gapped_amino_acid_dictionary
from utilities.tensor_utilities import get_device, print_shape


class FeatureExtractor:
    """
    Todo add class description
    Todo add device type
    Todo add dtype type
    """

    def __init__(self, file_path: str, number_cluster_sequences: int, mask_probability: float, device, dtype,
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
        self.number_cluster_sequences = number_cluster_sequences

        # Fetch all sequences in the msa file
        self.unprocessed_sequences = self.load_a3m_file()


        # Fetch MSA Sequence Tensor : Removing Duplications After 'insertion' removal
        # Fetch Deletion Count Keeping Track Of Consecutive Insertion On The Left
        self.msa_sequence_tensor, self.msa_deletion_count_tensor = self.compute_unique_sequences()

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

        # Masking Probability For More Robust Training
        self.mask_probability = mask_probability
        self.mask_cluster_centers(seed=self.seed)

        # Cluster Assignments For Extra MSA Sequences And Associated Assignment Counts On Input MSA Sequence
        self.assignments_tensor, self.assignments_count_tensor = self.assign_cluster()

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

        max_msa_clusters = min(self.number_cluster_sequences, self.total_sequences)

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
        self.input_msa_sequence_tensor = nn.functional.pad(self.input_msa_sequence_tensor, (0,1),value=0)
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

    # TODO Apply Cluster Averaging for deletion and profiling
    # - Must be properly explained

    # TODO Crop extra msa count

    # TODO Create Full MSA Feature

    # TODO Create Extra MSA Feature

    # TODO Get First Sequence
    # TODO Get One Hot Encoding
    # TODO Get Residue indexes

    # first_sequence = sequences[0]
    # target_feat = onehot_encode_aa_type(seq=first_sequence, include_gap_token=False)
    # residue_index = torch.arange(len(first_sequence))

    # variables of your object at the end.
    # return {
    #     'msa_feat': msa_feat,
    #     'extra_msa_feat': extra_msa_feat,
    #     'target_feat': target_feat,
    #     'residue_index': residue_index
    # }

    # Try to add a simple test with an input file.
    # Test without seed so random permutations are just not considered


features = FeatureExtractor(
    file_path="multiple_sequence_alignement.a3m",
    number_cluster_sequences=512,
    mask_probability=0.15,
    device=torch.device("cpu"),
    dtype=torch.float64,
    seed=0)
