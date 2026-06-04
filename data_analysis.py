from utilities.data.structure import Structure
from utilities.os_utilities import read_json

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

# Note : everything above can be deleted

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

