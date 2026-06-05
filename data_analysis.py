from utilities.os_utilities import read_json
from utilities.data.structure import Structure
from typing import Dict, List, Any
import math

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
        for k,v in entry.items():
            print(f"Key : {k}")
            print(v)
        exit()
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

