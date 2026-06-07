import os
from utilities.analysis_utilities import analyze_folder, analyze_summary_json

def main():
    json_file = "/home/robert_kelevra/Data/protein_data/openfold_processed_targets/open_fold_structure_data_summary.json"
    folder_to_analyze ="/home/robert_kelevra/Data/protein_data/openfold_processed_targets/structures"
    # 1. Check if summary JSON exists to avoid hours of processing
    if os.path.exists(json_file):
        print(f"--- Found existing summary: {json_file}. Analyzing... ---")
        analyze_summary_json(json_file, threshold=20.0)
    else:
        print(f"--- Summary not found. Generating from {folder_to_analyze}... ---")
        # Generate the JSON (flat format)
        analyze_folder(folder_to_analyze, output_file=json_file)
        # Then analyze it
        analyze_summary_json(json_file, threshold=20.0)

if __name__ == "__main__":
    main()
