import os
import torch
from pathlib import Path
import sys

# Add the project root to sys.path so we can import utilities
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utilities.data.input import ModelInput
from utilities.os_utilities import load_configuration, read_json

def precompute(number_of_samples: int = 5):
    # Setup paths
    data_folder = Path("data_examples/openfold")
    split_file = Path("dataset_splits/Train_small.json")
    config_file = Path("configurations/tiny_configuration.yaml")
    output_dir = Path("sandbox/precomputed_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_configuration(config_file)
    phase_config = config["TrainDataConfiguration"]

    # Load split
    cluster_mapping = read_json(str(split_file))
    protein_ids = []
    for members in cluster_mapping.values():
        protein_ids.extend(members)

    print(f"Precomputing {len(protein_ids)} proteins with {number_of_samples} samples each...")

    for protein_id in protein_ids:
        structure_path = data_folder / "structures" / f"{protein_id}.npz"
        record_path = data_folder / "records" / f"{protein_id}.json"
        msa_path = data_folder / "raw_msa" / f"{protein_id}.a3m"

        if not structure_path.exists() or not record_path.exists() or not msa_path.exists():
            print(f"Warning: Structure file not found for {protein_id}. Skipping.")
            continue

        print(f"Processing {protein_id}...")

        # Initialize the ModelInput once per protein to avoid redundant A3M parsing if possible
        # However, internal state like unique sequences is computed in __init__
        model_input = ModelInput(
            structure_path=str(structure_path),
            msa_path=str(msa_path),
            record_path=str(record_path),
            acceptance_slope_start=phase_config['acceptance_slope_start'],
            acceptance_slope_end=phase_config['acceptance_slope_end'],
            residue_crop_size=phase_config['residue_crop_size'],
            emphasize_beginning_crops=phase_config['emphasize_beginning_crops'],
            distribution_threshold=phase_config['distribution_threshold'],
            maximum_cluster_sequences=phase_config['maximum_cluster_sequences'],
            maximum_extra_msa_sequences=phase_config['maximum_extra_msa_sequences'],
            mask_probability=phase_config['mask_probability'],
            device=torch.device("cpu"),
            dtype=torch.float32)

        for i in range(number_of_samples):
            # Vary the seed for each sample to get different crops/masks
            batch_data = model_input.get_data(
                number_samples=phase_config['number_recycle_cycles'],
                seed=42 + i, 
                batch_mode=False )

            output_path = output_dir / f"{protein_id}_sample_{i}.pt"
            torch.save(batch_data, output_path)
        
        print(f"Saved {number_of_samples} samples for {protein_id}")

if __name__ == "__main__":
    precompute(number_of_samples=5)
