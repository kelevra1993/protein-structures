import torch
import unittest
from pathlib import Path
from utilities.data.input import ModelInput
from utilities.constants import (rigid_group_atom_position_map, atom_to_index,
                                 index_to_xxx)

current_directory = Path(__file__).parent
reference_directory = current_directory / "reference_values"


class TestInput(unittest.TestCase):
    def setUp(self):
        self.npz_path = reference_directory / "structures" / "P90561.npz"
        self.msa_path = reference_directory / "raw_msa" / "P90561.a3m"
        self.record_path = reference_directory / "records" / "P90561.json"
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        # Standard training-like configuration
        self.training_config = {
            "acceptance_slope_start": 256,
            "acceptance_slope_end": 512,
            "residue_crop_size": 128,
            "emphasize_beginning_crops": True,
            "distribution_threshold": 90,
            "maximum_cluster_sequences": 50,
            "maximum_extra_msa_sequences": 100,
            "mask_probability": 0.15
        }

        self.model_input = ModelInput(
            structure_path=str(self.npz_path),
            msa_path=str(self.msa_path),
            record_path=str(self.record_path),
            device=self.device,
            dtype=self.dtype,
            **self.training_config
        )

    def test_initialization(self):
        """Check if ModelInput is correctly initialized and sequences are loaded."""
        self.assertEqual(len(self.model_input.global_target_sequence), 354)
        self.assertEqual(self.model_input.global_amino_acid_sequence_labels.shape, (354,))
        self.assertTrue(self.model_input.global_msa_sequence_tensor.shape[1] == 354)

    def test_crop_indices(self):
        """Verify that crop indices are within bounds and have correct length."""
        for _ in range(10):
            start_index, end_index = self.model_input.get_crop_indices()
            self.assertTrue(0 <= start_index < end_index <= 354)
            self.assertEqual(end_index - start_index, 128)

    def test_get_data_shapes(self):
        """Check shapes of data returned by get_data with multiple recycle cycles."""
        number_recycle_samples = 3
        recycled_input_data = self.model_input.get_data(
            number_samples=number_recycle_samples, seed=42, batch_mode=False)

        # List of critical feature keys
        feature_keys = [
            "input_msa_feature", "input_extra_msa_feature", "input_sequence_feature",
            "input_residue_index_feature", "sequence_labels", "ground_truth_global_positions",
            "ground_truth_local_positions", "ground_truth_frames", "ground_truth_angles"
        ]
        for key in feature_keys:
            self.assertIn(key, recycled_input_data)
            # The last dimension must represent the recycle cycles
            self.assertEqual(recycled_input_data[key].shape[-1], number_recycle_samples)

        # Check spatial and feature dimensions (crop_size = 128)
        self.assertEqual(recycled_input_data["input_sequence_feature"].shape, (128, 21, number_recycle_samples))
        self.assertEqual(recycled_input_data["ground_truth_global_positions"].shape,
                         (128, 37, 3, number_recycle_samples))
        self.assertEqual(recycled_input_data["ground_truth_frames"].shape, (128, 8, 4, 4, number_recycle_samples))

    def test_batch_mode_formatting(self):
        """Check if batch_mode correctly prepends a batch dimension to all tensors."""
        batched_input_data = self.model_input.get_data(number_samples=1, seed=42, batch_mode=True)
        # Check first dimension (Batch)
        self.assertEqual(batched_input_data["input_sequence_feature"].shape[0], 1)
        # Full shape check: (Batch=1, Residues=128, Atoms=21, Cycles=1)
        self.assertEqual(batched_input_data["input_sequence_feature"].shape, (1, 128, 21, 1))

    def test_full_length_determinism_parity(self):
        """
        Integration Parity Test: Verify that full-length data without masking 
        matches the pre-computed reference tensors.
        """
        reference_file_path = reference_directory / "P90561_full_length_reference.pt"
        if not reference_file_path.exists():
            self.skipTest(f"Reference file {reference_file_path} not found.")

        deterministic_config = self.training_config.copy()
        deterministic_config["residue_crop_size"] = None
        deterministic_config["mask_probability"] = 0.0
        deterministic_config["maximum_cluster_sequences"] = 32
        deterministic_config["maximum_extra_msa_sequences"] = 64

        deterministic_input = ModelInput(
            structure_path=str(self.npz_path),
            msa_path=str(self.msa_path),
            record_path=str(self.record_path),
            device=self.device,
            dtype=self.dtype,
            **deterministic_config
        )

        # Generate data with same seed as reference generator
        generated_data = deterministic_input.get_data(number_samples=1, seed=0, batch_mode=False)
        reference_data = torch.load(reference_file_path, weights_only=True)

        # Compare critical tensors
        comparison_keys = ["input_sequence_feature", "sequence_labels",
                           "ground_truth_global_positions", "ground_truth_frames"]

        for key in comparison_keys:
            torch.testing.assert_close(generated_data[key], reference_data[key],
                                       atol=1e-4, rtol=1e-5,
                                       msg=f"Parity check failed for tensor: {key}")

    def test_cropped_local_coordinate_precision(self):
        """
        Verify that even after cropping, the local coordinates match 
        canonical positions within the 0.01 delta.
        """
        # Get one crop with recycling=1
        cropped_data = self.model_input.get_data(number_samples=1, seed=123, batch_mode=False)

        # Shape: (number_residues_crop, 37, 3, 1)
        local_positions = cropped_data["ground_truth_local_positions"].squeeze(-1)
        # We need the actual names of the cropped residues
        # Residue labels are in "sequence_labels" but we need the 3-letter codes
        sequence_labels = cropped_data["sequence_labels"].squeeze(-1)

        for residue_index in range(local_positions.shape[0]):
            amino_acid_label = int(sequence_labels[residue_index].item())
            residue_name = index_to_xxx[amino_acid_label]

            if residue_name == "UNK":
                continue  # Skip UNK for exact precision checks if needed, but it has mean values

            canonical_map = rigid_group_atom_position_map[residue_name]

            for atom_name, canonical_position in canonical_map.items():
                atom_idx = atom_to_index[atom_name]
                computed_position = local_positions[residue_index, atom_idx]

                torch.testing.assert_close(
                    computed_position,
                    canonical_position.to(device=self.device, dtype=self.dtype),
                    atol=0.01, rtol=0.01,
                    msg=f"Precision failure in crop at Residue {residue_index} ({residue_name}), Atom {atom_name}"
                )


if __name__ == "__main__":
    unittest.main()
