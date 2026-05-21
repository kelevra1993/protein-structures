import torch
import unittest
import os
from pathlib import Path
from feature_extraction.extractor import FeatureExtractor

current_directory = Path(__file__).parent
reference_directory = current_directory / "reference_values"


class TestFeatureExtractorIntegration(unittest.TestCase):
    def setUp(self):
        # Path to the sample MSA file moved to the tests directory
        self.msa_file_path = current_directory / "multiple_sequence_alignement.a3m"

        # Reference file paths (to be provided by the user)
        self.ref_msa_path = reference_directory / "input_msa_feature.pt"
        self.ref_extra_msa_path = reference_directory / "input_extra_msa_feature.pt"

        # Initialize the extractor with fixed parameters and seed for determinism
        self.extractor = FeatureExtractor(
            file_path=str(self.msa_file_path),
            maximum_cluster_sequences=512,
            maximum_extra_msa_sequences=5120,
            mask_probability=0.15,
            device=torch.device("cpu"),
            dtype=torch.float64,
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