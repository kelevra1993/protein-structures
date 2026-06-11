import argparse
from pathlib import Path
from utilities.data.data_splitter.data_splitting_utilities import (
    prepare_mmseqs_input,
    run_mmseqs_clustering,
    load_cluster_mapping,
    split_data_by_clusters
)

def main():
    parser = argparse.ArgumentParser(description="End-to-end data processing and splitting pipeline.")
    parser.add_argument("--a3m_folder", type=str, required=True, help="Path to the folder containing .a3m files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the generated output files (FASTA, TSV, JSONs).")
    parser.add_argument("--min_identity", type=float, default=0.4, help="Minimum sequence identity threshold for clustering (default: 0.4).")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of clusters to assign to the training set (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42).")

    args = parser.parse_args()

    a3m_folder = Path(args.a3m_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fasta_path = output_folder / "all_sequences.fasta"
    cluster_prefix = output_folder / "clusters"

    # Step 1: Prepare MMseqs2 input
    print("\n--- Step 1: Preparing MMseqs2 Input ---")
    prepare_mmseqs_input(str(a3m_folder), str(fasta_path))

    # Step 2: Run MMseqs2 Clustering
    print("\n--- Step 2: Running MMseqs2 Clustering ---")

    try:
        run_mmseqs_clustering(str(fasta_path), str(cluster_prefix), min_identity=args.min_identity)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure MMseqs2 is installed and available in your PATH to continue.")
        return

    # Step 3: Parse Clustering Results
    print("\n--- Step 3: Parsing Clustering Results ---")
    tsv_path = str(cluster_prefix) + "_cluster.tsv"
    if not Path(tsv_path).exists():
         print(f"Error: Clustering output TSV not found at {tsv_path}.")
         return
         
    cluster_mapping = load_cluster_mapping(tsv_path)

    # Step 4: Split Data
    print("\n--- Step 4: Splitting Data ---")
    split_data_by_clusters(cluster_mapping, str(output_folder), train_ratio=args.train_ratio, seed=args.seed)

    print(f"\nPipeline complete. Outputs saved in: {output_folder}")

if __name__ == "__main__":
    main()
