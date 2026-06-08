import torch
import unittest
from pathlib import Path
from utilities.data.dataloader import ProteinDataset, protein_collate_fn, get_protein_dataloader

current_directory = Path(__file__).parent
reference_directory = current_directory / "reference_values"


class TestDataloader(unittest.TestCase):
    def setUp(self):
        # Path configuration using permanent reference data structure
        self.data_folder = reference_directory
        self.split_file_path = current_directory / "test_split.json"
        
        self.dataloader_config = {
            "acceptance_slope_start": 256,
            "acceptance_slope_end": 512,
            "residue_crop_size": 64,
            "emphasize_beginning_crops": True,
            "distribution_threshold": 95,
            "maximum_cluster_sequences": 32,
            "maximum_extra_msa_sequences": 64,
            "mask_probability": 0.15,
            "number_recycle_cycles": 2,
            "use_single_representative": False
        }

    def test_dataset_initialization(self):
        """Check if ProteinDataset correctly maps protein IDs from the split file."""
        protein_dataset = ProteinDataset(
            data_folder=str(self.data_folder),
            split_file_path=str(self.split_file_path),
            device=torch.device("cpu"),
            dtype=torch.float32,
            **self.dataloader_config
        )
        self.assertEqual(len(protein_dataset), 1)
        self.assertEqual(protein_dataset.protein_ids, ["P90561"])

    def test_dataset_getitem_content(self):
        """Verify that __getitem__ returns a valid data dictionary with expected tensor shapes."""
        protein_dataset = ProteinDataset(
            data_folder=str(self.data_folder),
            split_file_path=str(self.split_file_path),
            device=torch.device("cpu"),
            dtype=torch.float32,
            **self.dataloader_config
        )
        first_sample_data = protein_dataset[0]
        self.assertIsInstance(first_sample_data, dict)
        # 2 recycle cycles, 21 AA types, 64 residue crop
        self.assertEqual(first_sample_data["input_sequence_feature"].shape, (64, 21, 2))

    def test_collate_fn_fast_path(self):
        """Test the optimized fast path for batch size 1 in the collate function."""
        mock_protein_data = {
            "tensor_a": torch.randn(10, 5),
            "tensor_b": torch.randn(3, 4, 2)
        }
        single_item_batch = [mock_protein_data]
        collated_batch = protein_collate_fn(single_item_batch)
        
        # Verify batch dimension was added correctly
        self.assertEqual(collated_batch["tensor_a"].shape, (1, 10, 5))
        self.assertEqual(collated_batch["tensor_b"].shape, (1, 3, 4, 2))
        torch.testing.assert_close(collated_batch["tensor_a"][0], mock_protein_data["tensor_a"])

    def test_collate_fn_dynamic_padding(self):
        """Test if collate function correctly pads multiple proteins of different lengths."""
        first_protein_item = {"data": torch.ones(10, 5)}
        second_protein_item = {"data": torch.ones(15, 5) * 2}
        multi_protein_batch = [first_protein_item, second_protein_item]
        
        collated_batch = protein_collate_fn(multi_protein_batch)
        # Max length is 15
        self.assertEqual(collated_batch["data"].shape, (2, 15, 5))
        
        # Verify zero-padding for the shorter sequence
        self.assertEqual(collated_batch["data"][0, 12, 0], 0.0)
        # Verify data integrity for the longer sequence
        self.assertEqual(collated_batch["data"][1, 14, 0], 2.0)

    def test_dataloader_factory_iteration(self):
        """Verify the get_protein_dataloader factory produces a functional, iterable DataLoader."""
        protein_dataloader = get_protein_dataloader(
            data_folder=str(self.data_folder),
            split_file_path=str(self.split_file_path),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            **self.dataloader_config
        )
        self.assertIsInstance(protein_dataloader, torch.utils.data.DataLoader)
        
        # Verify we can successfully iterate and retrieve a batch
        for batch_data in protein_dataloader:
            self.assertEqual(batch_data["input_sequence_feature"].shape, (1, 64, 21, 2))
            break

if __name__ == "__main__":
    unittest.main()
