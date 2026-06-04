import torch
import unittest
import os
from pathlib import Path
from feature_extraction.extractor import FeatureExtractor
from utilities.data.msa import load_a3m_file, compute_unique_sequences

current_directory = Path(__file__).parent
reference_directory = current_directory / "reference_values"


class TestFeatureExtractorIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the sample MSA file moved to the tests directory
        self.msa_file_path = current_directory / "multiple_sequence_alignement.a3m"

        # Reference file paths (to be provided by the user)
        self.ref_msa_path = reference_directory / "input_msa_feature.pt"
        self.ref_extra_msa_path = reference_directory / "input_extra_msa_feature.pt"

        device = torch.device("cpu")
        dtype = torch.float64

        # 1. Pre-process MSA Data
        unprocessed_sequences = load_a3m_file(str(self.msa_file_path))
        target_sequence = unprocessed_sequences[0]
        global_msa_sequence_tensor, global_msa_deletion_count_tensor = compute_unique_sequences(
            unprocessed_sequences=unprocessed_sequences,
            device=device,
            dtype=dtype
        )

        # 2. Initialize the extractor with pre-computed globals
        self.extractor = FeatureExtractor(
            target_sequence=target_sequence,
            global_msa_sequence_tensor=global_msa_sequence_tensor,
            global_msa_deletion_count_tensor=global_msa_deletion_count_tensor,
            maximum_cluster_sequences=512,
            maximum_extra_msa_sequences=5120,
            mask_probability=0.15,
            device=device,
            dtype=dtype,
            seed=0
        )

    def test_msa_feature_parity(self):
        """Checks if the generated input_msa_feature matches the reference tensor exactly."""
        if not os.path.exists(self.ref_msa_path):
            self.skipTest(f"Reference file {self.ref_msa_path} not found.")

        reference_msa_feature = torch.load(self.ref_msa_path, weights_only=True)

        # Using allclose for floating point comparison, or equal if exact bit-parity is expected
        torch.testing.assert_close(self.extractor.input_msa_feature, reference_msa_feature)

    def test_extra_msa_feature_parity(self):
        """Checks if the generated input_extra_msa_feature matches the reference tensor exactly."""
        if not os.path.exists(self.ref_extra_msa_path):
            self.skipTest(f"Reference file {self.ref_extra_msa_path} not found.")

        reference_extra_msa_feature = torch.load(self.ref_extra_msa_path, weights_only=True)

        torch.testing.assert_close(self.extractor.input_extra_msa_feature, reference_extra_msa_feature)


if __name__ == "__main__":
    unittest.main()
