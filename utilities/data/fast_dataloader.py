import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utilities.data.dataloader import protein_collate_fn
from utilities.os_utilities import read_json

import random

class PrecomputedProteinDataset(Dataset):
    def __init__(self, precomputed_dir: str, split_file_path: str, phase: str, number_samples: int = 5):
        self.precomputed_dir = Path(precomputed_dir) / phase
        self.split_file_path = Path(split_file_path)
        self.number_samples = number_samples
        self.phase = phase

        # Load split
        cluster_mapping = read_json(str(self.split_file_path))
        self.protein_ids = []
        for members in cluster_mapping.values():
            self.protein_ids.extend(members)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> dict:
        protein_id = self.protein_ids[index]
        
        # Randomly select one of the precomputed samples if in Train phase, otherwise just take 0
        if self.phase == "Train":
            sample_index = random.randint(0, self.number_samples - 1)
        else:
            sample_index = 0
            
        file_path = self.precomputed_dir / f"{protein_id}_sample_{sample_index}.pt"
        
        return torch.load(file_path)

def get_fast_dataloader(precomputed_dir: str, split_file_path: str, phase: str, number_samples: int = 5, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    dataset = PrecomputedProteinDataset(precomputed_dir=precomputed_dir, split_file_path=split_file_path, phase=phase, number_samples=number_samples)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=protein_collate_fn,
        drop_last=False
    )
    return dataloader
