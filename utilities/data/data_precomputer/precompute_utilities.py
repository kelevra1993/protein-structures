"""File used for precomputation of protein data to speed up training"""
import torch
from pathlib import Path
from tqdm import tqdm

from utilities.data.input import ModelInput
from utilities.os_utilities import load_configuration, load_experiment_configuration, read_json, print_red, print_green


def precompute_dataset(experiment_configuration_path: str, output_directory: str, number_samples: int):
    """
    Precomputes the dataset features into .pt files for Train, Validation, and Test phases.
    
    Args:
        experiment_configuration_path: Path to the experiment configuration YAML file.
        output_directory: Base directory where the precomputed data will be stored.
        number_samples: Number of random variations to generate per protein.
    """
    experiment_configuration = load_experiment_configuration(experiment_configuration_path)

    model_configuration_path = experiment_configuration.get("configuration_path")
    if not model_configuration_path:
        raise ValueError("Key 'configuration_path' is missing from the experiment configuration.")

    full_configuration = load_configuration(model_configuration_path)

    data_folder = Path(experiment_configuration["data_folder"])
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing is done on multiple datasets ranging from training, validation and testing.
    # Note that for testing we should load the full uncropped sequence
    # Try to think of a process where we might want to actually have mutliple crop sizes
    phases = [("Train", experiment_configuration.get("train_split_file")),
              ("Validation", experiment_configuration.get("validation_split_file")),
              ("Test", experiment_configuration.get("test_split_file"))]

    for phase_name, split_file in phases:
        # if phase file does not exit it should not break.
        # Typically if we have no test data
        if not split_file or str(split_file) == "":
            print(f"Skipping {phase_name} phase: no split file provided in configuration.")
            continue

        # Create the Train, Validation and Test Folder that will contained preprocessed data.
        phase_output_directory = output_directory_path / phase_name
        phase_output_directory.mkdir(parents=True, exist_ok=True)

        phase_configuration_key = f"{phase_name}DataConfiguration"
        if phase_configuration_key not in full_configuration:
            print(f"Skipping {phase_name} phase: '{phase_configuration_key}' not found in configuration.")
            continue

        phase_configuration = full_configuration[phase_configuration_key]

        # Only considering Train, Validation and Test Files that were picked.
        split_path = Path(split_file)
        if not split_path.exists():
            print(f"Warning: Split file {split_path} not found. Skipping {phase_name} phase.")
            continue

        cluster_mapping = read_json(str(split_path))
        protein_ids = []

        # Get all proteins that were separated by cluster
        for members in cluster_mapping.values():
            protein_ids.extend(members)

        print(f"\n--- Precomputing {phase_name} Phase ({len(protein_ids)} proteins, {number_samples} samples each) ---")

        # Keep Track Of Skipped protein ids
        skipped_proteins = []
        processed_proteins = []

        for protein_id in tqdm(protein_ids, desc=f"Processing {phase_name} proteins"):
            structure_path = data_folder / "structures" / f"{protein_id}.npz"
            record_path = data_folder / "records" / f"{protein_id}.json"
            msa_path = data_folder / "raw_msa" / f"{protein_id}.a3m"

            if not all([structure_path.exists(), record_path.exists(), msa_path.exists()]):
                skipped_proteins.append(protein_id)
                continue

            try:
                model_input = ModelInput(
                    structure_path=str(structure_path),
                    msa_path=str(msa_path),
                    record_path=str(record_path),
                    acceptance_slope_start=phase_configuration['acceptance_slope_start'],
                    acceptance_slope_end=phase_configuration['acceptance_slope_end'],
                    residue_crop_size=phase_configuration['residue_crop_size'],
                    emphasize_beginning_crops=phase_configuration['emphasize_beginning_crops'],
                    distribution_threshold=phase_configuration['distribution_threshold'],
                    maximum_cluster_sequences=phase_configuration['maximum_cluster_sequences'],
                    maximum_extra_msa_sequences=phase_configuration['maximum_extra_msa_sequences'],
                    mask_probability=phase_configuration['mask_probability'],
                    device=torch.device("cpu"),
                    dtype=experiment_configuration.get("dtype", torch.float32))

                for sample_index in range(number_samples):
                    # We vary the seed to ensure different random crops/masks per sample
                    batch_data = model_input.get_data(
                        number_samples=phase_configuration['number_recycle_cycles'],
                        seed=42 + sample_index, batch_mode=False)

                    output_path = phase_output_directory / f"{protein_id}_sample_{sample_index}.pt"
                    torch.save(batch_data, output_path)
                processed_proteins.append(protein_id)
            except KeyboardInterrupt:
                exit("KeyboardInterrupt From User, Exiting Script")
            except Exception as e:
                # We do not stop the processing
                skipped_proteins.append(protein_id)
                continue

        print_green(f"For Phase {phase_name}, We Processed {len(processed_proteins)} Proteins")
        print_red(f"For Phase {phase_name}, We Skipped {len(skipped_proteins)} Proteins")
