"""File used to precompute data so that data retrieval bottleneck can be improved"""
import os
import argparse

from pathlib import Path
from utilities.data.data_precomputer.precompute_utilities import precompute_dataset
from utilities.os_utilities import print_blue, globalise_path


def main():
    parser = argparse.ArgumentParser(description="Precompute PyTorch dataset tensors for faster training.")
    parser.add_argument("--configuration_path", type=str, required=True,
                        help="Path to the experiment configuration YAML file.")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Path to save the precomputed datasets.")
    parser.add_argument("--number_samples", type=int, default=5,
                        help="Number of random variations to compute per protein.")
    parser.add_argument("--number_workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers for precomputation.")

    args = parser.parse_args()

    # Check if paths exists if not try to make them absolute and try again
    project_folder = Path(os.getcwd()).parents[2]

    args.configuration_path = globalise_path(
        absolute_parent_path=project_folder,
        target_path=Path(args.configuration_path))

    args.output_directory = globalise_path(
        absolute_parent_path=project_folder,
        target_path=Path(args.output_directory))

    # Precompute datasets
    precompute_dataset(experiment_configuration_path=str(args.configuration_path),
                       output_directory=str(args.output_directory),
                       number_samples=args.number_samples,
                       number_workers=args.number_workers)

    print(f"\nPrecomputation complete. Tensors saved to {args.output_directory}")


if __name__ == "__main__":
    main()
