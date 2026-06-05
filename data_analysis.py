import time
import numpy as np

np.set_printoptions(linewidth=1000, threshold=np.inf)
from utilities.tensor_utilities import print_tensor_list

start = time.time()
from utilities.os_utilities import read_json
from utilities.data.structure import Structure
from utilities.data.input import ModelInput
from typing import Dict, List, Any
import math

# Get the model input object
structure_path = "data_examples/openfold/structures/P90561.npz"
msa_path = "data_examples/openfold/raw_msa/P90561.a3m"
record_path = "data_examples/openfold/records/P90561.json"
print(f"This Took {time.time() - start} seconds")
start = time.time()
model_input = ModelInput(
    structure_path=structure_path,
    msa_path=msa_path,
    record_path=record_path,
    maximum_cluster_sequences=25,
    maximum_extra_msa_sequences=50,
)
print(f"This Took {time.time() - start} seconds")
print(model_input.keep_input())

training_data = model_input.get_data(number_samples=4,
                                     random_samples=True,
                                     crop_size=None,
                                     seed=None,
                                     batch_mode=False,
                                     emphasize_beginning_crops=False)







exit()
training_data = model_input.get_data(number_samples=4,
                                     random_samples=True,
                                     crop_size=None,
                                     seed=None,
                                     batch_mode=False,
                                     emphasize_beginning_crops=False)

for key, value in training_data.items():
    print(f"{key=}")
    print(f" - Shape : {list(value.shape)}")

    match key:
        case 'input_msa_feature':
            # [batch, msa_sequence, crop_size_number_residues, 49, number_cyles]
            # Example : [1, 25, 10, 49, 4]
            print(40 * '-')
            print_tensor_list(tensor=value[..., 0, 1, :, 0], round=2)
            print_tensor_list(tensor=value[..., 0, 1, :, 1], round=2)
            print_tensor_list(tensor=value[..., 0, 1, :, 2], round=2)
            print(40*'-')
        case 'input_extra_msa_feature':
            # [batch, extra_msa_sequence, crop_size_number_residues, 25, number_cyles]
            # Example : [1, 25, 50, 49, 4]
            print(40 * '-')
            print_tensor_list(tensor=value[..., 0, 1, :, 0], round=2)
            print_tensor_list(tensor=value[..., 0, 1, :, 1], round=2)
            print_tensor_list(tensor=value[..., 0, 1, :, 2], round=2)
            print(40 * '-')
        case 'input_sequence_feature':
            # [batch, crop_size_number_residues, 21, number_cyles]
            # Example : [1, 50, 21, 4]
            print(40 * '-')
            print_tensor_list(tensor=value[..., 0, :, 0], round=2)
            print_tensor_list(tensor=value[..., 0, :, 1], round=2)
            print_tensor_list(tensor=value[..., 0, :, 2], round=2)
            print_tensor_list(tensor=value[..., 0, :, 3], round=2)
            print(40 * '-')
        case 'input_residue_index_feature':
            # [batch, crop_size_number_residues, 21, number_cyles]
            # Example : [1, 10, 4]
            print(40 * '-')
            print_tensor_list(tensor=value[..., :, 0], round=2)
            print_tensor_list(tensor=value[..., :, 1], round=2)
            print_tensor_list(tensor=value[..., :, 2], round=2)
            print_tensor_list(tensor=value[..., :, 3], round=2)
            print(40 * '-')

exit()


def analyze_manifest(file_path: str):
    """
    Analyzes the manifest JSON file and prints statistics.

    Args:
        file_path: Path to the manifest.json file.
    """
    print(f"--- Analyzing Manifest: {file_path} ---")
    data = read_json(file_path)
    total_structures = len(data)

    if total_structures == 0:
        print("Empty manifest.")
        return

    method_counts = {}
    resolution_bins = {"0-1": 0, "1-2": 0, "2-3": 0, "3-4": 0, "4-5": 0, "5+": 0}
    chain_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    mol_type_counts = {}

    for entry in data:

        structure_info = entry.get("structure", {})

        # 1. Method statistics
        method = structure_info.get("method")
        method_str = str(method)
        method_counts[method_str] = method_counts.get(method_str, 0) + 1

        # 2. Resolution statistics
        res = structure_info.get("resolution")
        if res is not None:
            if 0 <= res < 1:
                resolution_bins["0-1"] += 1
            elif 1 <= res < 2:
                resolution_bins["1-2"] += 1
            elif 2 <= res < 3:
                resolution_bins["2-3"] += 1
            elif 3 <= res < 4:
                resolution_bins["3-4"] += 1
            elif 4 <= res < 5:
                resolution_bins["4-5"] += 1
            elif res >= 5:
                resolution_bins["5+"] += 1

        # 3. Number of chains statistics
        num_chains = structure_info.get("num_chains", 0)
        if num_chains == 1:
            chain_counts["1"] += 1
        elif num_chains == 2:
            chain_counts["2"] += 1
        elif num_chains == 3:
            chain_counts["3"] += 1
        elif num_chains == 4:
            chain_counts["4"] += 1
        elif num_chains >= 5:
            chain_counts["5+"] += 1

        # 4. Mol type statistics (per chain)
        chains = entry.get("chains", [])
        for chain in chains:
            m_type = str(chain.get("mol_type"))
            mol_type_counts[m_type] = mol_type_counts.get(m_type, 0) + 1

    # Print Methods
    print("\n--- Experimental Methods ---")
    for method, count in method_counts.items():
        percentage = (count / total_structures) * 100
        print(f"Method: {method:20} | Count: {count:6} | Percentage: {percentage:6.2f}%")

    # Print Resolution
    print("\n--- Resolution Ranges (Angstroms) ---")
    for r_range, count in resolution_bins.items():
        print(f"Range: {r_range:10} | Count: {count:6}")

    # Print Chain Counts
    print("\n--- Number of Chains per Structure ---")
    for c_count, count in chain_counts.items():
        print(f"Chains: {c_count:10} | Count: {count:6}")

    # Print Mol Types
    print("\n--- Molecular Types (Total across all chains) ---")
    total_chains = sum(mol_type_counts.values())
    for m_type, count in mol_type_counts.items():
        percentage = (count / total_chains) * 100 if total_chains > 0 else 0
        print(f"Type: {m_type:20} | Count: {count:6} | Percentage: {percentage:6.2f}%")


if __name__ == "__main__":
    # Run analysis on the example
    # analyze_manifest("manifest_example.json")

    # Run analysis on the RCSB manifest
    analyze_manifest("data_examples/rcsb/manifest.json")

    # Run analysis on the OpenFold manifest
    analyze_manifest("data_examples/openfold/manifest.json")

# Note : everything above can be deleted
exit()
# Load the structure object
structure_file = "data_examples/openfold/structures/P90561.npz"
structure = Structure(structure_file)

# Inspect components
print(f"--- Loaded Structure: {structure_file} ---")
print(f"Atoms: {len(structure.atoms)}")
print(f"Residues: {len(structure.residues)}")
print(f"Chains: {len(structure.chains)}")

# Example inspect
print("\n--- Example Residue (with custom amino_acid_index) ---")
print(structure.residues[0])

exit()
# Get manifest data
manifest_file = "data_examples/openfold/manifest.json"
# manifest_file = "manifest_example.json"
manifest_data = read_json(manifest_file)
print(manifest_data)

for data_information in manifest_data:
    data_id = data_information["id"]
    data_structure = data_information["structure"]
    data_chains = data_information["chains"]

    # The following below are currently unknown
    data_interfaces = data_information["interfaces"]
    data_affinity = data_information["affinity"]
    data_md = data_information["md"]

    # Data Preprocessing steps
    # Get the resolution
    resolution = data_structure["resolution"]

    # Get the method by which the structure was obtained ?
    # cryo-em / nmr / x-ray crystallography ?
    method = data_structure["method"]

    # Get the number of chains
    # In our case make life easier by just taking elements that have one chain?
    number_chains = data_structure["num_chains"]

    # Number interfaces ?
    number_interfaces = data_structure["num_interfaces"]
