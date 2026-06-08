import random
import torch
from typing import Optional, Tuple, List, Dict
from collections import Counter
from utilities.data.structure import Structure, Residue, Atom
from utilities.data.msa import load_a3m_file, compute_unique_sequences, one_hot_encode_amino_acid_types
from feature_extraction.extractor import FeatureExtractor
from utilities.constants import x_to_xxx, all_amino_acid_dictionary
from utilities.geometry_utilities import (create_alternative_truth_positions,
                                          create_alternative_truth_angles,
                                          create_alternative_truth_transformation_matrix)


class ModelInput:
    """
    Represents a single training or inference example, containing both the
    physical structure and associated MSA data, alongside training-specific metadata.
    """

    def __init__(self,
                 structure_path: str,
                 msa_path: str,
                 acceptance_slope_start: int,
                 acceptance_slope_end: int,
                 residue_crop_size: int | None,
                 emphasize_beginning_crops: bool,
                 distribution_threshold: int,
                 maximum_cluster_sequences: int,
                 maximum_extra_msa_sequences: int,
                 mask_probability: float,
                 record_path: Optional[str] = None,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        Initializes a ModelInput object, representing a single protein complex for training or inference.

        This class serves as the primary data container, integrating structural information from NPZ files
        with multiple sequence alignment (MSA) data. It handles sequence cropping, acceptance probability
        calculations for data balancing, and feature extraction preparation.

        Args:
            structure_path (str): File path to the .npz file containing the protein's physical structure.
            msa_path (str): File path to the .a3m file containing the multiple sequence alignment.
            record_path (str, optional): File path to the JSON record containing structure metadata
                (e.g., resolution, method).
            acceptance_slope_start (int): Minimum protein length for the acceptance probability ramp.
            acceptance_slope_end (int): Maximum protein length for the acceptance probability ramp.
            residue_crop_size (int | None): The number of residues to include in a crop. If None,
                no cropping is performed.
            emphasize_beginning_crops (bool): If True, applies a stochastic bias to sample crops closer to the N-terminus.
            distribution_threshold (int): Percentage threshold for filtering out sequences with
                unusually high concentrations of a single amino acid.
            maximum_cluster_sequences (int): Maximum number of unique sequences to retain for MSA clustering.
            maximum_extra_msa_sequences (int): Maximum number of sequences to retain for the extra MSA stack.
            mask_probability (float): Probability of masking a residue during MSA feature extraction.
            device (torch.device): The target device for tensor computations.
            dtype (torch.dtype): The target data type for floating-point tensors.
        """
        # Set simple requirements
        self.device = device
        self.dtype = dtype

        # Load the core physical structure
        self.structure = Structure(npz_path=structure_path, record_path=record_path,
                                   device=self.device, dtype=self.dtype)

        # Training specific parameters
        self.acceptance_slope_start = acceptance_slope_start
        self.acceptance_slope_end = acceptance_slope_end
        self.residue_crop_size = residue_crop_size
        self.emphasize_beginning_crops = emphasize_beginning_crops
        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences
        self.mask_probability = mask_probability

        # Calculate acceptance probability immediately
        self.acceptance_probability = self._compute_acceptance_probability()

        # MSA Data (Mandatory)
        self.msa_path = msa_path

        self.unprocessed_sequences = load_a3m_file(self.msa_path)

        # The target sequence is always the first one
        self.global_target_sequence = self.unprocessed_sequences[0]

        # Compute the global amino acid sequence labels
        sequence_labels_list = [all_amino_acid_dictionary.get(aa, 20) for aa in self.global_target_sequence]
        self.global_amino_acid_sequence_labels = torch.tensor(sequence_labels_list, device=self.device,
                                                              dtype=torch.long)

        self.global_msa_sequence_tensor, self.global_msa_deletion_count_tensor = compute_unique_sequences(
            unprocessed_sequences=self.unprocessed_sequences,
            device=self.device, dtype=self.dtype)

        # Full amino-acid distribution of the target sequence
        self.amino_acid_distribution = self._compute_amino_acid_distribution()
        self.distribution_check = all(v < distribution_threshold for v in self.amino_acid_distribution.values())

    def _compute_acceptance_probability(self) -> float:
        """
        Calculates the probability of accepting this input for training based on its size.
        The probability is calculated as:
        (1 / acceptance_slope_end) * (max(min(number_residues, acceptance_slope_end), acceptance_slope_start))
        By default probability of 0.5 for sequences lower that acceptance_slope_start ,
        1.0 for sequences above acceptance_slope_end and number_residues/acceptance_slope_end in between
        """
        number_residues = self.structure.number_residues

        probability = (1.0 / self.acceptance_slope_end) * (
            max(min(number_residues, self.acceptance_slope_end), self.acceptance_slope_start)
        )
        return probability

    def _compute_amino_acid_distribution(self) -> Dict[str, float]:
        """
        Computes the amino acid percentage distribution of the target sequence.

        This method analyzes the primary protein sequence (the first entry in the MSA)
        to determine the relative frequency of each amino acid residue. This distribution
        is primarily used to identify and filter out low-complexity sequences or
        unusually biased proteins (e.g., highly repetitive regions) during the
        data selection process for training.

        Returns:
            Dict[str, float]: A dictionary where keys are 3-letter amino acid codes
                (standardized via x_to_xxx) and values are their corresponding
                percentage frequencies (0.0 to 100.0), rounded to two decimal places.
        """

        distribution_counter = Counter(self.global_target_sequence)
        distribution = {x_to_xxx.get(x, "UNK"): round(100 * count / len(self.global_target_sequence), 2) for x, count in
                        distribution_counter.items()}

        return distribution

    def keep_input(self) -> bool:
        """
        Determines whether to keep this input based on its acceptance probability.
        Useful for stochastic filtering of training samples in a loop.

        Returns:
            bool: True if the input is accepted, False otherwise.
        """
        return random.random() < self.acceptance_probability

    def get_crop_indices(self) -> tuple[int, int]:
        """
        Generates random start and end indices for cropping the sequence and structure.

        This method is invoked only when `residue_crop_size` is specified. It provides
        the indices necessary to slice MSA and structural tensors into a fixed-size window.

        Returns:
            tuple[int, int]: A pair of (start_index, end_index) for slicing.
        """
        num_residues = self.structure.number_residues

        # If the protein is smaller than the crop size, return the whole thing
        if num_residues <= self.residue_crop_size:
            return 0, num_residues

        # Maximum possible start index to ensure we don't go out of bounds
        max_start_index = num_residues - self.residue_crop_size

        if not self.emphasize_beginning_crops:
            # Uniform sampling
            start_index = random.randint(0, max_start_index)
        else:
            # Bias sampling towards the beginning
            start_emphasis = random.randint(0, self.residue_crop_size)
            max_start_emp = max(0, max_start_index - start_emphasis)
            start_index = random.randint(0, max_start_emp)

        end_index = start_index + self.residue_crop_size
        return start_index, end_index

    def get_cropped_msa_data(self, start_index: int, end_index: int) -> Tuple[
        str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the MSA data cropped to the specified indices.

        Args:
            start_index (int): Start of the crop.
            end_index (int): End of the crop.

        Returns:
            Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
                - The cropped target sequence (str).
                - The cropped global_amino_acid_sequence_labels.
                - The cropped global_msa_sequence_tensor.
                - The cropped global_msa_deletion_count_tensor.
        """
        if self.unprocessed_sequences is None:
            raise ValueError("MSA data has not been loaded. Initialize ModelInput with an msa_path.")

        cropped_target_sequence = self.global_target_sequence[start_index:end_index]

        # Slice tensors along the residue dimension
        cropped_sequence_labels = self.global_amino_acid_sequence_labels[start_index:end_index]
        cropped_sequence_tensor = self.global_msa_sequence_tensor[:, start_index:end_index, :]
        cropped_deletion_tensor = self.global_msa_deletion_count_tensor[:, start_index:end_index]

        return cropped_target_sequence, cropped_sequence_labels, cropped_sequence_tensor, cropped_deletion_tensor

    def get_cropped_structure_data(self, start_index: int, end_index: int) -> Tuple[List[Residue], List[Atom]]:
        """
        Retrieves the residues and atoms within the specified crop indices.

        Args:
            start_index (int): Start residue index.
            end_index (int): End residue index.

        Returns:
            Tuple[List[Residue], List[Atom]]: The cropped residues and atoms.
        """
        cropped_residues = self.structure.residues[start_index:end_index]

        if not cropped_residues:
            return [], []

        # Get the range of atoms associated with these residues
        atom_start = cropped_residues[0].atom_start_index
        last_residue = cropped_residues[-1]
        atom_end = last_residue.atom_start_index + last_residue.atom_count

        cropped_atoms = self.structure.atoms[atom_start:atom_end]

        return cropped_residues, cropped_atoms

    def get_cropped_ground_truth_data(self, start_index: int, end_index: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the ground truth tensors cropped to the specified indices.

        Args:
            start_index (int): Start of the crop.
            end_index (int): End of the crop.

        Returns:
            Tuple[torch.Tensor, ...]: The cropped global positions, local positions, frames, and angles.
        """
        # Slice tensors along the residue dimension (dim=0)
        cropped_global_positions = (
            self.structure.ground_truth_global_positions[start_index:end_index].to(self.device, dtype=self.dtype))

        cropped_local_positions = (
            self.structure.ground_truth_local_positions[start_index:end_index].to(self.device, dtype=self.dtype))

        cropped_frames = self.structure.ground_truth_frames[start_index:end_index].to(self.device, dtype=self.dtype)

        cropped_angles = self.structure.ground_truth_angles[start_index:end_index].to(self.device, dtype=self.dtype)

        return cropped_global_positions, cropped_local_positions, cropped_frames, cropped_angles

    def get_data(self, number_samples: int,
                 seed: Optional[int] = None, batch_mode: bool = False) -> Dict[str, torch.Tensor]:
        """
        Constructs a comprehensive input dictionary for the model, supporting recycling cycles.

        This method orchestrates cropping (if enabled), feature extraction for each recycling
        cycle, and the duplication of ground truth data to match the recycle dimension.
        All features are stacked along a final 'cycle' dimension.

        Args:
            number_samples (int): The number of recycling cycles (iterations) to generate data for.
            seed (Optional[int]): Base random seed for reproducible stochastic features (e.g., masking).
            batch_mode (bool): If True, prepends a batch dimension of size 1 to all tensors.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of stacked features. All tensors follow the
                (..., number_samples) shape, or (1, ..., number_samples) if batch_mode is True.
                - input_msa_feature: (number_clusters, number_residues, msa_feature_dimension, number_samples)
                - input_extra_msa_feature: (number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_samples)
                - input_sequence_feature: (number_residues, input_sequence_feature_dimension, number_samples)
                - input_residue_index_feature: (number_residues, number_samples)
                - sequence_labels: (number_residues, number_samples)
                - ground_truth_global_positions: (number_residues, 37, 3, number_samples)
                - ground_truth_local_positions: (number_residues, 37, 3, number_samples)
                - ground_truth_frames: (number_residues, 8, 4, 4, number_samples)
                - ground_truth_angles: (number_residues, 7, 2, number_samples)
                - alternative_ground_truth_global_positions: (number_residues, 37, 3, number_samples)
                - alternative_ground_truth_local_positions: (number_residues, 37, 3, number_samples)
                - alternative_ground_truth_frames: (number_residues, 8, 4, 4, number_samples)
                - alternative_ground_truth_angles: (number_residues, 7, 2, number_samples)
        """

        # Determine the residue range (Cropping)
        if self.residue_crop_size:
            start_index, end_index = self.get_crop_indices()
        else:
            start_index, end_index = 0, self.structure.number_residues

        # Extract cropped raw MSA data
        target_sequence, sequence_labels, msa_sequence_tensor, msa_deletion_tensor = self.get_cropped_msa_data(
            start_index=start_index, end_index=end_index)

        # Extract cropped ground truth data
        (ground_truth_global_positions,
         ground_truth_local_positions,
         ground_truth_frames,
         ground_truth_angles) = self.get_cropped_ground_truth_data(start_index=start_index,
                                                                   end_index=end_index)

        # Compute alternative ground truths by adding a temporary batch dimension
        batched_labels = sequence_labels.unsqueeze(0)

        alternative_ground_truth_global_positions = create_alternative_truth_positions(
            ground_truth_positions=ground_truth_global_positions.unsqueeze(0),
            sequence_amino_acid_labels=batched_labels).squeeze(0)

        alternative_ground_truth_local_positions = create_alternative_truth_positions(
            ground_truth_positions=ground_truth_local_positions.unsqueeze(0),
            sequence_amino_acid_labels=batched_labels).squeeze(0)

        alternative_ground_truth_frames = create_alternative_truth_transformation_matrix(
            transformation_matrix=ground_truth_frames.unsqueeze(0),
            sequence_amino_acid_labels=batched_labels).squeeze(0)

        alternative_ground_truth_angles = create_alternative_truth_angles(
            ground_truth_angles=ground_truth_angles.unsqueeze(0),
            sequence_amino_acid_labels=batched_labels).squeeze(0)

        # For training, we recycle data by shuffling the msa data
        cycle_data = {"input_msa_feature": [], "input_extra_msa_feature": [],
                      "input_sequence_feature": [], "input_residue_index_feature": [],
                      "sequence_labels": [],
                      "ground_truth_global_positions": [], "ground_truth_local_positions": [],
                      "ground_truth_frames": [], "ground_truth_angles": [],
                      "alternative_ground_truth_global_positions": [],
                      "alternative_ground_truth_local_positions": [],
                      "alternative_ground_truth_frames": [],
                      "alternative_ground_truth_angles": []}

        # Generate features for each cycle (number_samples)
        # We wrap this in no_grad to prevent accidental gradient tracking during data prep
        with torch.no_grad():

            # Precompute static features outside the recycling loop to avoid redundant calculations
            precomputed_input_sequence_feature = one_hot_encode_amino_acid_types(
                sequence=target_sequence, include_gap_token=False,
                device=msa_sequence_tensor.device, dtype=msa_sequence_tensor.dtype)

            # Residue indices
            precomputed_input_residue_index_feature = torch.arange(len(target_sequence),
                                                                   device=msa_sequence_tensor.device)

            # Amino acid distribution : used for masking
            precomputed_total_amino_acid_distribution = msa_sequence_tensor.mean(dim=0, keepdim=True)

            for cycle in range(number_samples):
                # Vary seed per cycle to allow different random masks/shuffles
                current_seed = seed + cycle if seed is not None else None

                extractor = FeatureExtractor(
                    target_sequence=target_sequence,
                    global_msa_sequence_tensor=msa_sequence_tensor,
                    global_msa_deletion_count_tensor=msa_deletion_tensor,
                    input_sequence_feature=precomputed_input_sequence_feature,
                    input_residue_index_feature=precomputed_input_residue_index_feature,
                    total_amino_acid_distribution=precomputed_total_amino_acid_distribution,
                    maximum_cluster_sequences=self.maximum_cluster_sequences,
                    maximum_extra_msa_sequences=self.maximum_extra_msa_sequences,
                    mask_probability=self.mask_probability,
                    device=msa_sequence_tensor.device,
                    dtype=msa_sequence_tensor.dtype,
                    seed=current_seed)

                # Offset residue indices to be absolute (protein-relative)
                absolute_residue_indices = extractor.input_residue_index_feature + start_index

                cycle_data["input_msa_feature"].append(extractor.input_msa_feature)
                cycle_data["input_extra_msa_feature"].append(extractor.input_extra_msa_feature)
                cycle_data["input_sequence_feature"].append(extractor.input_sequence_feature)
                cycle_data["input_residue_index_feature"].append(absolute_residue_indices)

                # Ground truth data and labels are identical across cycles, duplicate to match recycle dimension
                cycle_data["sequence_labels"].append(sequence_labels)
                cycle_data["ground_truth_global_positions"].append(ground_truth_global_positions)
                cycle_data["ground_truth_local_positions"].append(ground_truth_local_positions)
                cycle_data["ground_truth_frames"].append(ground_truth_frames)
                cycle_data["ground_truth_angles"].append(ground_truth_angles)

                # Add alternative truth data, based on symmetries.
                cycle_data["alternative_ground_truth_global_positions"].append(
                    alternative_ground_truth_global_positions)
                cycle_data["alternative_ground_truth_local_positions"].append(alternative_ground_truth_local_positions)
                cycle_data["alternative_ground_truth_frames"].append(alternative_ground_truth_frames)
                cycle_data["alternative_ground_truth_angles"].append(alternative_ground_truth_angles)

        # Stack along the last dimension (cycle dimension)
        batch_input_dictionary = {key: torch.stack(values, dim=-1) for key, values in cycle_data.items()}

        # Optional Batch Unsqueeze (for testing/single-sample inference)
        if batch_mode:
            batch_input_dictionary = {key: value.unsqueeze(0) for key, value in batch_input_dictionary.items()}

        return batch_input_dictionary
