"""File used for precomputation of protein data to speed up training"""
import torch
import json
from pathlib import Path
from tqdm import tqdm

from utilities.data.input import ModelInput
from utilities.os_utilities import load_experiment_configuration, read_json, print_red, print_green, print_blue

from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_protein(protein_id: str,
                           data_folder: Path,
                           phase_output_directory: Path,
                           phase_configuration: dict,
                           number_samples: int,
                           experiment_configuration: dict) -> tuple[str, bool]:
    """
    Processes a single protein: initializes ModelInput, generates samples, and saves to .pt files.

    Returns:
        tuple[str, bool]: (protein_id, success_status)
    """
    structure_path = data_folder / "structures" / f"{protein_id}.npz"
    record_path = data_folder / "records" / f"{protein_id}.json"
    msa_path = data_folder / "raw_msa" / f"{protein_id}.a3m"

    if not all([structure_path.exists(), record_path.exists(), msa_path.exists()]):
        return protein_id, False

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

        return protein_id, True
    except Exception:
        return protein_id, False


def precompute_dataset(experiment_configuration_path: str, output_directory: str, number_samples: int,
                       number_workers: int = 10):
    """
    Precomputes the dataset features into .pt files for Train, Validation, and Test phases.
    
    Args:
        experiment_configuration_path: Path to the experiment configuration YAML file.
        output_directory: Base directory where the precomputed data will be stored.
        number_samples: Number of random variations to generate per protein.
        number_workers: Number of parallel workers for precomputation.
    """
    experiment_configuration, full_configuration = load_experiment_configuration(experiment_configuration_path)

    data_folder = Path(experiment_configuration["data_folder"])
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing is done on multiple datasets ranging from training, validation and testing.
    phases = [("Train", experiment_configuration.get("train_split_file")),
              ("Validation", experiment_configuration.get("validation_split_file")),
              ("Test", experiment_configuration.get("test_split_file"))]

    for phase_name, split_file in phases:
        # if phase file does not exit it should not break.
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

        json_tracker_path = phase_output_directory / f"{phase_name}_precomputed_samples.json"
        processed_proteins = []

        if json_tracker_path.exists():
            print_blue(f"Json Tracker {json_tracker_path} Exists, Fetching Precomputed Data...")
            processed_proteins = read_json(path=str(json_tracker_path))

        # Filter out already processed proteins
        remaining_protein_ids = [pid for pid in protein_ids if pid not in processed_proteins]
        skipped_proteins = []

        if not remaining_protein_ids:
            print_green(f"All {len(protein_ids)} proteins for {phase_name} are already precomputed.")
            continue

        try:
            with ProcessPoolExecutor(max_workers=number_workers) as executor:
                futures = {executor.submit(process_single_protein,
                                           protein_id,
                                           data_folder,
                                           phase_output_directory,
                                           phase_configuration,
                                           number_samples,
                                           experiment_configuration): protein_id
                           for protein_id in remaining_protein_ids}

                for future in tqdm(as_completed(futures),
                                   total=len(remaining_protein_ids),
                                   desc=f"Processing {phase_name} proteins"):
                    protein_id, success = future.result()
                    if success:
                        processed_proteins.append(protein_id)
                    else:
                        skipped_proteins.append(protein_id)

        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt From User, shutting down workers and saving progress...")
            # Note: Executor shutdown with wait=False is handled by the 'with' block context,
            #  but we want to save what we have so far.
        finally:
            with open(json_tracker_path, "w") as f:
                json.dump(processed_proteins, f, indent=4)
            print_green(f"For Phase {phase_name}, We have a total of {len(processed_proteins)} Precomputed Proteins")
            print_red(f"For Phase {phase_name}, We Skipped {len(skipped_proteins)} Proteins during this run")
