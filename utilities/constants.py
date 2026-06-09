"""
File that contains all constants that will be used in the project
"""
import torch
import numpy as np

# No printing of scientific notations
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=500, threshold=np.inf)

# TODO ADD THREE LETTER ENCODING EASIER FOR FUTURE WORK
# TODO CONSIDER ADDING INTEGER INDICES TO AMINO ACID RESIDUES DIRECTLY ?

# Canonical Amino Acid Residues (20 amino acids)
# Created by taking 3-letter AA codes and sorting them alphabetically.
canonical_amino_acid_residues = ["A", "R",
                                 "N", "D",
                                 "C", "Q",
                                 "E", "G",
                                 "H", "I",
                                 "L", "K",
                                 "M", "F",
                                 "P", "S",
                                 "T", "W",
                                 "Y", "V"]

# Include Unknown Amino Acid as 'X' : Used For Input Sequence Feature
# Including amino acids like selenocysteine, Pyrrolysine, ...e.t.c
all_amino_acid_residues = canonical_amino_acid_residues + ["X"]

# Including Gaps
gapped_amino_acid_residues = all_amino_acid_residues + ["-"]

# Turn them into dictionaries
# Todo : Give a simple example for both
all_amino_acid_dictionary = {k: index for index, k in enumerate(all_amino_acid_residues)}
gapped_amino_acid_dictionary = {k: index for index, k in enumerate(gapped_amino_acid_residues)}

# Atoms positions relative to the 8 rigid groups/bases/frames, defined by the pre-omega, phi,
# psi and chi angles:
# 0: 'backbone group',
# 1: 'pre-omega-group', (empty)
# 2: 'phi-group', (currently empty, because it defines only hydrogens)
# 3: 'psi-group',
# 4,5,6,7: 'chi1,2,3,4-group'
# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_index/base_index/frame_index, rel_position]
rigid_group_atom_positions = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
    ],
    "UNK": [
        ["N", 0, (-0.526, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.771, -1.214)],
        ["O", 3, (0.626, 1.062, 0.000)],
    ],
}

# Todo add small description + example
# Conversion of the positions to tensors
rigid_group_atom_positions = {
    key: [
        [name, index, torch.tensor(pos)]
        for name, index, pos in values
    ] for key, values in rigid_group_atom_positions.items()

}

# Todo add small description + example
# Structure:
# rigid_group_atom_position_map = {
#   'ARG': {
#      'N': torch.tensor(-0.524, 1.362, -0.000),
#      'CA': torch.tensor(0.000, 0.000, 0.000),
#       ...
#   }
#   ...
# }
# Todo add small description + example
rigid_group_atom_position_map = {
    aa: {
        entry[0]: entry[2] for entry in entries
    } for aa, entries in rigid_group_atom_positions.items()
}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array). -> 37 Atoms.
atom_types = ["N", "CA", "C", "CB", "O",
              "CG", "CG1", "CG2",
              "OG", "OG1",
              "SG",
              "CD", "CD1", "CD2",
              "ND1", "ND2",
              "OD1", "OD2",
              "SD",
              "CE", "CE1", "CE2", "CE3",
              "NE", "NE1", "NE2",
              "OE1", "OE2",
              "CH2",
              "NH1", "NH2",
              "OH",
              "CZ", "CZ2", "CZ3",
              "NZ",
              "OXT"]

# Maps atoms to their indices
# Todo add small description + example
atom_to_index = {atom_type: i for i, atom_type in enumerate(atom_types)}
index_to_atom = {i: atom_type for i, atom_type in enumerate(atom_types)}
number_atom_types = len(atom_types)  # := 37.

# Todo add small description + example
atom_local_positions = torch.zeros((21, 37, 3))

# Todo add small description + example
atom_frame_indices = torch.zeros((21, 37), dtype=torch.int64)

# Todo add small description + example
atom_mask = torch.zeros((21, 37)).to(torch.bool)

for i, (aa, values) in enumerate(rigid_group_atom_positions.items()):
    for name, index, pos in values:
        atom_local_positions[i, atom_to_index[name]] = pos
        atom_frame_indices[i, atom_to_index[name]] = index
        atom_mask[i, atom_to_index[name]] = True

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order (see below).
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

# Non-chi coordinate frames centers (consistent across all residues)
# 0: backbone, 1: pre-omega, 2: phi, 3: psi
non_chi_frame_centers = ["CA", "CA", "N", "C"]

# Side chain coordinate frames are constructed
# by using this order of side-chain atoms as their centers
chi_angles_frame_centers = {
    'ALA': [],
    'ARG': ['CB', 'CG', 'CD', 'NE'],
    'ASN': ['CB', 'CG'],
    'ASP': ['CB', 'CG'],
    'CYS': ['CB'],
    'GLN': ['CB', 'CG', 'CD'],
    'GLU': ['CB', 'CG', 'CD'],
    'GLY': [],
    'HIS': ['CB', 'CG'],
    'ILE': ['CB', 'CG1'],
    'LEU': ['CB', 'CG'],
    'LYS': ['CB', 'CG', 'CD', 'CE'],
    'MET': ['CB', 'CG', 'SD'],
    'PHE': ['CB', 'CG'],
    'PRO': ['CB', 'CG'],
    'SER': ['CB'],
    'THR': ['CB'],
    'TRP': ['CB', 'CG'],
    'TYR': ['CB', 'CG'],
    'VAL': ['CB'],
    'UNK': [],
}

# This dictionary is used to compute the different dihedral/torsion angles of ground truths for chi frames.
chi_dihedral_dictionary = {
    "ARG": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD', 'atom_3': 'NE', 'atom_4': 'CZ'},
    "ASN": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'OD1'},
    "ASP": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'OD1'},
    "CYS": {'atom_0': 'CB', 'atom_1': 'SG'},
    "GLN": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD', 'atom_3': 'OE1'},
    "GLU": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD', 'atom_3': 'OE1'},
    "HIS": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'ND1'},
    "ILE": {'atom_0': 'CB', 'atom_1': 'CG1', 'atom_2': 'CD1'},
    "LEU": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD1'},
    "LYS": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD', 'atom_3': 'CE', 'atom_4': 'NZ'},
    "MET": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'SD', 'atom_3': 'CE'},
    "PHE": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD1'},
    "PRO": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD'},
    "SER": {'atom_0': 'CB', 'atom_1': 'OG'},
    "THR": {'atom_0': 'CB', 'atom_1': 'OG1'},
    "TRP": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD1'},
    "TYR": {'atom_0': 'CB', 'atom_1': 'CG', 'atom_2': 'CD1'},
    "VAL": {'atom_0': 'CB', 'atom_1': 'CG1'},
    "UNK": {},
}
#########################################################
# Management of alternative truths for loss computation #
#########################################################
# Simple dictionaries to retrieve canonical amino acid indices
# Three Letter Codes
xxx_to_index = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
                'UNK': 20}
index_to_xxx = {value: key for key, value in xxx_to_index.items()}

# Single Letter Codes
x_to_index = {key: index for index, key in enumerate(all_amino_acid_residues)}
index_to_x = {index: key for index, key in enumerate(all_amino_acid_residues)}

# Single <-> Three Letter Code Changes
x_to_xxx = {x: index_to_xxx[index] for x, index in x_to_index.items()}
xxx_to_x = {xxx: x for x, xxx in x_to_xxx.items()}

# Angles : shape (21, 7, 2)
alternative_angle_mask = torch.ones((21, 7, 2))

# Rigid Groups with atom symetry based on torsion angles
# For amino-acids such as aspartic acid, glutamic acid, phenylalanine and tyrosine
angle_symetry_amino_acids = {
    "ASP": 4,  # Chi2
    "GLU": 5,  # Chi3
    "PHE": 4,  # Chi2
    "TYR": 4,  # Chi2
}

for amino_acid, chi_angle_index in angle_symetry_amino_acids.items():
    alternative_angle_mask[xxx_to_index[amino_acid], chi_angle_index] *= -1

# Positions : shape (21, 37)
alternative_position_mask = torch.arange(number_atom_types).repeat(21, 1)

# Swapped atoms
position_symetry_atoms = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}

for amino_acid, atoms_to_swap in position_symetry_atoms.items():
    amino_acid_index = xxx_to_index[amino_acid]

    for atom_a, atom_b in atoms_to_swap:
        atom_a_index = atom_to_index[atom_a]
        atom_b_index = atom_to_index[atom_b]
        alternative_position_mask[amino_acid_index, atom_a_index] = atom_b_index
        alternative_position_mask[amino_acid_index, atom_b_index] = atom_a_index

# Variable containing atoms that might have ambiguous positions
ambiguous_position_mask = torch.abs(alternative_position_mask - torch.arange(number_atom_types)).to(dtype=torch.bool)

# # todo to be kept for now for testing then removed later
# import numpy as np
# np.set_printoptions(linewidth=200, threshold=np.inf)
# print(alternative_position_mask.numpy())
# print(ambiguous_position_mask.numpy())
# print(alternative_position_mask.numpy() - np.array(list(range(37))))
# print(ambigous_position_mask.numpy())
# for i in range(21):
#     print(f"Amino Acid : {index_to_xxx[i]}")
#     # print(angle_alternative_truth_mask[i].numpy())
#     print(position_alternative_mask[i].numpy())
#     print(position_alternative_mask[i].numpy()-np.array(list(range(37))))
#     print(30 * '-')
