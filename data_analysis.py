import numpy as np
from sympy import sturm
from tqdm import tqdm
from typing import Dict
from utilities.os_utilities import read_json, read_npz_file

# Understanding Structure Data
structure_file = "data_examples/openfold/structures/P90561.npz"
structure_data = read_npz_file(path=structure_file)
structure_data_keys = list(structure_data.keys())

print(f"NPZ Keys: {structure_data_keys}")

for key in structure_data_keys:
    print(f"Main Key :: {key.upper()} :: {structure_data[key].shape}")

for key in structure_data_keys:
    print(f"Main Key :: {key.upper()}")
    print(structure_data[key][:500])
    print(structure_data["coords"][:5])

    exit()
    if isinstance(structure_data[key], dict):
        sub_keys = list(structure_data[key].keys())
        print(f" - Keys For {key} Are : {structure_data_keys}")

# print(structure_data['atoms'])

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

