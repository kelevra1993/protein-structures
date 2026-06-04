import numpy as np

# Dictionary to map the integer encoding back to atom string names.
# 46 is mapped to "N" based on your first data point (Atomic number 7).
ATOM_TYPE_MAP = {
    46: "N",
    # 47: "CA", etc... populate this based on your dataset's residue_constants
}


def decode_atom_name(encoded_array):
    """
    todo to be reviewed
    Decodes the integer array into a human-readable atom string.
    Extracts the first element, assuming the trailing zeros are padding.
    """
    if isinstance(encoded_array, (list, tuple, np.ndarray)) and len(encoded_array) > 0:
        primary_id = int(encoded_array[0])
        return ATOM_TYPE_MAP.get(primary_id, f"UNKNOWN_ATOM_{primary_id}")
    return "UNKNOWN_FORMAT"


def humanize_npz_structure_data(key, data):
    """
    todo to be reviewed
    Translates raw NPZ tuple/array data into a human-readable dictionary.
    """
    if key == 'atoms' or key == 'atom':
        return {
            'atom_name': decode_atom_name(data[0]),
            'atomic_number': data[1],
            'formal_charge': data[2],
            'coordinates': data[3],
            'unknown_vector_4': data[4],
            'atom_mask': data[5],
            'residue_or_chain_index': data[6]
        }

    # Generic fallback for keys we have not analyzed yet
    humanized_dict = {}
    for i, item in enumerate(data):
        humanized_dict[f"unknown_{key}_field_{i}"] = item

    return humanized_dict


# # --- Example Usage ---
# if __name__ == "__main__":
#     raw_atom_data = (
#         [46, 0, 0, 0],
#         7,
#         0,
#         [55.924, 44.204, -60.962],
#         [1.8903918, -1.5252995, -0.42638594],
#         True,
#         0
#     )
#
#     result = humanize_npz_structure_data('atoms', raw_atom_data)
#
#     for k, v in result.items():
#         print(f"{k}: {v}")