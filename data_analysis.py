import numpy as np
from tqdm import tqdm
from dataclasses import asdict
from utilities.data_utilities import humanize_npz_structure_data, humanize_atom, humanize_residue, humanize_chain
from utilities.os_utilities import read_json, read_npz_file

# Understanding Structure Data
structure_file = "data_examples/openfold/structures/P90561.npz"
structure_data = read_npz_file(path=structure_file)
structure_data_keys = list(structure_data.keys())

# NPZ Keys: ['atoms', 'bonds', 'residues', 'chains', 'connections', 'interfaces', 'mask', 'coords', 'ensemble']
# Find the examples below for one of our examples
# NPZ Key Atoms of shape (2951,)
# NPZ Key Bonds of shape (0,)
# NPZ Key Residues of shape (354,)
# NPZ Key Chains of shape (1,)
# NPZ Key Connections of shape (0,)
# NPZ Key Interfaces of shape (0,)
# NPZ Key Mask of shape (1,)
# NPZ Key Coords of shape (2951,)
# NPZ Key Ensemble of shape (1,)
# print(f"NPZ Keys: {structure_data_keys}")

# # Inspect the first 2 atoms
print("--- ATOMS ---")
raw_atoms = structure_data['atoms'][:500]
Atoms = [humanize_atom(row) for row in raw_atoms]
# for i, raw_atom in enumerate(raw_atoms):
#     human_atom = humanize_atom(raw_atom)
#     print(f"Atom {i}: {human_atom}")
# exit()

# Inspect the first 2 residues
print("\n--- RESIDUES ---")
raw_residues = structure_data['residues'][:50]
Residues = [humanize_residue(row) for row in raw_residues]
# for i, raw_residue in enumerate(raw_residues):
#     human_res = humanize_residue(raw_residue)
#     # print(Atoms[human_res.pseudo_carbon_beta_atom_index])
#     print(f"Residue {i}: {human_res}")
# exit()

# Inspect the first chain
print("\n--- CHAINS ---")
raw_chains = structure_data['chains']
for i, raw_chain in enumerate(raw_chains):
    human_chain = humanize_chain(raw_chain)
    print(f"Chain {i}: {human_chain}")

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

