"""
Todo add description
"""


# todo to be determined
# from utilities.constants import

class FeatureExtractor():
    """
    Todo add class description
    Todo add device type
    Todo add dtype type
    """

    def __init__(self, file_path: str, number_cluster_sequences: int, mask_probability: float, device, dtype):
        """
        # Add documentation
        :param file_path:
        """

        self.file_path = file_path
        self.device = device
        self.dtype = dtype

        # Add number residues as soon as possible
        # Add number total sequences as soon as possible

    # Todo Implement loading of a3m file.

    # Todo Implement Onehot encoder with gap option.

    # Todo Compute Unique Sequences (outputs unique sequences + deletion matrix )
    # - properly document deletion count

    # Todo Compute AA Distribution
    # Only used for masking and nothing else

    # Todo Select Cluster Sequences
    # - Yields four outputs msa_cluster, msa_deletion_count and their 'extra_' counterparts

    # Todo Apply Masking
    # - See how to call the masked / augmented input sequence feature
    # - Or Just call the initial one un_alterered_input_sequence_feature ?
    # - Try to see why we would need the un_altered_input_sequence_feature to begin with ?
    # - If not useful, just get rid of it

    # TODO Run Assignment Counts

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
    # Test without seed so randompermutations are just not considered
