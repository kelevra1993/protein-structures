import random
import torch
from typing import Optional, Tuple, List, Dict
from collections import Counter
from utilities.data.structure import Structure, Residue, Atom
from utilities.data.msa import load_a3m_file, compute_unique_sequences
from feature_extraction.extractor import FeatureExtractor
from utilities.tensor_utilities import get_device
from utilities.constants import x_to_xxx


class ModelInput:
    """
    Represents a single training or inference example, containing both the 
    physical structure and associated MSA data, alongside training-specific metadata.
    """

    def __init__(self,
                 structure_path: str,
                 msa_path: str,
                 record_path: Optional[str] = None,
                 acceptance_slope_start: int = 256,
                 acceptance_slope_end: int = 512,
                 residue_crop_size: int = 256,
                 distribution_threshold: int = 80,
                 maximum_cluster_sequences: int = 512,
                 maximum_extra_msa_sequences: int = 5120,
                 mask_probability: float = 0.15,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):

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
        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences
        self.mask_probability = mask_probability

        # Calculate acceptance probability immediately
        self.acceptance_probability = self._compute_acceptance_probability()

        # MSA Data (Mandatory)
        self.msa_path = msa_path
        self.unprocessed_sequences: List[str] = [""]
        self.global_msa_sequence_tensor = None
        self.global_msa_deletion_count_tensor = None

        self.unprocessed_sequences = load_a3m_file(self.msa_path)
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

        target_sequence = self.unprocessed_sequences[0]

        distribution_counter = Counter(target_sequence)
        distribution = {x_to_xxx[x]: round(100 * count / len(target_sequence), 2) for x, count in
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

    def get_crop_indices(self, emphasize_beginning_crops: bool, crop_size: Optional[int] = None) -> tuple[int, int]:
        """
        Generates a start and end index used to crop the sequence and structure.

        Args:
            emphasize_beginning_crops (bool): If True, biases the start_index towards the beginning.
            crop_size (Optional[int]): Override the default residue_crop_size.

        Returns:
            tuple[int, int]: The start_index and end_index for cropping.
        """
        active_crop_size = crop_size if crop_size is not None else self.residue_crop_size
        num_residues = self.structure.number_residues

        # If the protein is smaller than the crop size, return the whole thing
        if num_residues <= active_crop_size:
            return 0, num_residues

        # Maximum possible start index to ensure we don't go out of bounds
        max_start_index = num_residues - active_crop_size

        if not emphasize_beginning_crops:
            # Uniform sampling
            start_index = random.randint(0, max_start_index)
        else:
            # Bias sampling towards the beginning
            start_emphasis = random.randint(0, active_crop_size)
            max_start_emp = max(0, max_start_index - start_emphasis)
            start_index = random.randint(0, max_start_emp)

        end_index = start_index + active_crop_size
        return start_index, end_index

    def get_cropped_msa_data(self, start_index: int, end_index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Retrieves the MSA data cropped to the specified indices.

        Args:
            start_index (int): Start of the crop.
            end_index (int): End of the crop.

        Returns:
            Tuple[str, torch.Tensor, torch.Tensor]:
                - The cropped target sequence (str).
                - The cropped global_msa_sequence_tensor.
                - The cropped global_msa_deletion_count_tensor.
        """
        if self.unprocessed_sequences is None:
            raise ValueError("MSA data has not been loaded. Initialize ModelInput with an msa_path.")

        # The target sequence is always the first one
        target_sequence = self.unprocessed_sequences[0]
        cropped_target_sequence = target_sequence[start_index:end_index]

        # Slice tensors along the residue dimension (dim=1)
        cropped_sequence_tensor = self.global_msa_sequence_tensor[:, start_index:end_index, :]
        cropped_deletion_tensor = self.global_msa_deletion_count_tensor[:, start_index:end_index]

        return cropped_target_sequence, cropped_sequence_tensor, cropped_deletion_tensor

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

    def get_data(self, number_samples: int, random_samples: bool = True, crop_size: Optional[int] = None,
                 seed: Optional[int] = None, batch_mode: bool = False,
                 emphasize_beginning_crops: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generates a batch input dictionary for the model's forward pass,
        supporting multi-cycle recycling features.

        Args:
            number_samples (int): Number of recycling cycles (iterations).
            random_samples (bool): If True, applies random cropping.
            crop_size (Optional[int]): Override default residue_crop_size.
            seed (Optional[int]): Base random seed for feature extraction.
            batch_mode (bool): If True, unsqueezes tensors to add a batch dimension of 1.
            emphasize_beginning_crops (bool): If True, biases random cropping towards 
                the start of the sequence.
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing input features ready for the Model.
        """
        # Determine the residue range (Cropping)
        if random_samples:
            start_index, end_index = self.get_crop_indices(emphasize_beginning_crops=emphasize_beginning_crops,
                                                           crop_size=crop_size)
        else:
            start_index, end_index = 0, self.structure.number_residues

        # Extract cropped raw MSA data
        target_sequence, msa_sequence_tensor, msa_deletion_tensor = self.get_cropped_msa_data(start_index=start_index,
                                                                                              end_index=end_index)

        # Extract cropped ground truth data
        (ground_truth_global_positions,
         ground_truth_local_positions,
         ground_truth_frames,
         ground_truth_angles) = self.get_cropped_ground_truth_data(start_index=start_index,
                                                                   end_index=end_index)

        # For training we recycle data by shuffling the msa data
        cycle_data = {"input_msa_feature": [], "input_extra_msa_feature": [],
                      "input_sequence_feature": [], "input_residue_index_feature": [],
                      "ground_truth_global_positions": [], "ground_truth_local_positions": [],
                      "ground_truth_frames": [], "ground_truth_angles": []}

        # Generate features for each cycle (number_samples)
        # We wrap this in no_grad to prevent accidental gradient tracking during data prep
        with torch.no_grad():
            for cycle in range(number_samples):
                # Vary seed per cycle to allow different random masks/shuffles
                current_seed = seed + cycle if seed is not None else None

                extractor = FeatureExtractor(
                    target_sequence=target_sequence,
                    global_msa_sequence_tensor=msa_sequence_tensor,
                    global_msa_deletion_count_tensor=msa_deletion_tensor,
                    maximum_cluster_sequences=self.maximum_cluster_sequences,
                    maximum_extra_msa_sequences=self.maximum_extra_msa_sequences,
                    mask_probability=self.mask_probability,
                    device=msa_sequence_tensor.device,
                    dtype=msa_sequence_tensor.dtype,
                    seed=current_seed
                )

                # Offset residue indices to be absolute (protein-relative)
                absolute_residue_indices = extractor.input_residue_index_feature + start_index

                cycle_data["input_msa_feature"].append(extractor.input_msa_feature)
                cycle_data["input_extra_msa_feature"].append(extractor.input_extra_msa_feature)
                cycle_data["input_sequence_feature"].append(extractor.input_sequence_feature)
                cycle_data["input_residue_index_feature"].append(absolute_residue_indices)

                # Ground truth data is identical across cycles, but we duplicate it to match the recycle dimension
                cycle_data["ground_truth_global_positions"].append(ground_truth_global_positions)
                cycle_data["ground_truth_local_positions"].append(ground_truth_local_positions)
                cycle_data["ground_truth_frames"].append(ground_truth_frames)
                cycle_data["ground_truth_angles"].append(ground_truth_angles)

        # Stack along the last dimension (cycle dimension)
        batch_input_dict = {key: torch.stack(values, dim=-1) for key, values in cycle_data.items()}

        # Optional Batch Unsqueeze (for testing/single-sample inference)
        if batch_mode:
            batch_input_dict = {key: value.unsqueeze(0) for key, value in batch_input_dict.items()}

        return batch_input_dict
