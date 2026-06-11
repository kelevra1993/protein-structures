import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from utilities.data.data_precomputer.precompute_utilities import precompute_dataset


def main():
    parser = argparse.ArgumentParser(description="Precompute PyTorch dataset tensors for faster training.")
    parser.add_argument("--configuration_path", type=str, required=True,
                        help="Path to the experiment configuration YAML file.")
    parser.add_argument("--output_directory", type=str, required=True,
                        help="Path to save the precomputed datasets.")
    parser.add_argument("--number_samples", type=int, default=5,
                        help="Number of random variations to compute per protein.")

    args = parser.parse_args()

    # Precompute datasets
    precompute_dataset(
        experiment_configuration_path=args.configuration_path,
        output_directory=args.output_directory,
        number_samples=args.number_samples
    )

    print(f"\nPrecomputation complete. Tensors saved to {args.output_directory}")


if __name__ == "__main__":
    main()
