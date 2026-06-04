import torch
from typing import List, Tuple
from utilities.constants import all_amino_acid_dictionary, gapped_amino_acid_dictionary


def load_a3m_file(file_path: str) -> List[str]:
    """
    Parses the .a3m file and extracts protein sequences.

    Lines starting with '>' are treated as headers and ignored. The following line
    is extracted as a sequence.

    :param file_path: Path to the .a3m file.
    :return: A list of sequence strings extracted from the file.
    """
    with open(file_path, "r") as msa_data:
        content = msa_data.readlines()
        sequences = [content[index + 1].strip() for index, line in enumerate(content) if line[0] == ">"]
    return sequences


def one_hot_encode_amino_acid_types(sequence: str, include_gap_token: bool, device: torch.device,
                                    dtype: torch.dtype) -> torch.Tensor:
    """
    Converts a sequence string into a one-hot encoded tensor.

    - If include_gap_token is False: Shape: (number_residues, number_canonical_amino_acids)
    - If include_gap_token is True: Shape: (number_residues, number_gapped_amino_acids)

    :param sequence: The amino acid sequence string to encode.
    :param include_gap_token: Whether to include the gap token '-' in the encoding categories.
    :param device: The target device.
    :param dtype: The target data type.
    :return: A one-hot encoded tensor of the sequence.
    """
    amino_acid_dictionary = all_amino_acid_dictionary if not include_gap_token else gapped_amino_acid_dictionary

    sequence_indices = torch.tensor([amino_acid_dictionary[amino_acid_index] for amino_acid_index in sequence],
                                    device=device)

    encoding = torch.nn.functional.one_hot(sequence_indices, num_classes=len(amino_acid_dictionary))
    return encoding.to(dtype)


def compute_unique_sequences(unprocessed_sequences: List[str],
                             device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Processes unprocessed sequences to remove duplicates and track insertions.

    Insertions (represented by lowercase letters in .a3m) are removed to maintain
    a consistent protein length across the MSA. The number of consecutive insertions
    to the left of each residue is tracked in a deletion count matrix.

    - unique_sequences_tensor shape: (total_sequences, number_residues, number_gapped_amino_acids)
    - deletion_count_matrix shape: (total_sequences, number_residues)

    :param unprocessed_sequences: List of raw sequences from the MSA file.
    :param device: The target device.
    :param dtype: The target data type.
    :return: A tuple containing the unique gapped sequence tensor and the deletion count tensor.
    """
    deletion_count_matrix = []
    unique_sequences = []

    for sequence in unprocessed_sequences:
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
    deletion_count_matrix_tensor = torch.tensor(deletion_count_matrix, dtype=dtype, device=device)

    unique_sequences_matrix = [
        one_hot_encode_amino_acid_types(sequence,
                                        include_gap_token=True,
                                        device=device, dtype=dtype) for sequence in unique_sequences]

    unique_sequences_tensor = torch.stack(unique_sequences_matrix, dim=0).to(device=device, dtype=dtype)

    return unique_sequences_tensor, deletion_count_matrix_tensor
