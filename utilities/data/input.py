import random
from typing import Optional
from utilities.data.structure import Structure


class ModelInput:
    """
    Represents a single training or inference example, containing both the 
    physical structure and associated MSA data, alongside training-specific metadata.
    """

    def __init__(self,
                 structure_path: str,
                 msa_path: Optional[str] = None,
                 record_path: Optional[str] = None,
                 acceptance_slope_start: int = 256,
                 acceptance_slope_end: int = 512,
                 residue_crop_size: int = 256):

        # Load the core physical structure
        self.structure = Structure(npz_path=structure_path, record_path=record_path)

        # Training specific parameters (to compute the probability of keeping a sequence)
        self.acceptance_slope_start = acceptance_slope_start
        self.acceptance_slope_end = acceptance_slope_end
        self.residue_crop_size = residue_crop_size

        # Calculate acceptance probability immediately
        self.acceptance_probability = self._compute_acceptance_probability()

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
