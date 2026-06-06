import torch
import numpy as np
from utilities.tensor_utilities import get_device

np.set_printoptions(linewidth=500, threshold=np.inf)

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list, specialised_one_hot_encoder
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss, \
    compute_local_distance_difference_test, compute_plddt_loss
from utilities.constants import alternative_angle_mask, alternative_position_mask, index_to_xxx, \
    ambiguous_position_mask, atom_types, rigid_group_atom_positions, rigid_group_atom_position_map, \
    chi_angles_frame_centers, chi_angles_mask
from utilities.geometry_utilities import create_alternative_truth_transformation_matrix, create_4x4_transform_matrix, \
    invert_4x4_transform_matrix, apply_transformation_on_vector, compute_dihedral_angle, \
    make_transformation_matrix_around_ex, create_3x3_rotation_matrix, \
    turn_quaternion_to_3x3_matrix, make_transformation_matrix_around_ex

from utilities.data.structure import Structure
from utilities.constants import atom_to_index, atom_frame_indices, chi_dihedral_dictionary


# For Testing / Debugging Comparison to constant.py. to see if we will ultimately just set everything to what is in constant.py
def frame_debugger(atom_position_dictionary, residue_name, frame_to_consider=None):
    for atom_name, atom_information in atom_position_dictionary.items():

        atom_frame = atom_information["frame"]
        current_atom_frame = atom_information["current_frame_used"]

        if frame_to_consider and frame_to_consider != atom_frame:
            continue

        if current_atom_frame == atom_frame:
            local_position = atom_information["frame_coordinates"].numpy().round(4)
            constant_position = rigid_group_atom_position_map[residue_name][atom_name].numpy().round(4)
            difference = local_position - constant_position
            difference_norm = torch.linalg.norm(torch.tensor(difference)).numpy()

            if difference_norm != 0.0:
                print(40 * '-')
                print(f"Local      {atom_name} : {local_position}")
                print(f"Consant    {atom_name} : {constant_position}")
                print(f"Delta      {atom_name} : {difference.round(4)}")
                print(f"Delta Norm {atom_name} : {difference_norm.round(4)}")
                print(40 * '-')


# compute backbones based on atom coordinates in structure file
device = torch.device("cpu")
dtype = torch.float64
# Step 1 : Get a structure object with a structure npz file
structure_object = Structure(npz_path="data_examples/openfold/structures/P90561.npz",
                             record_path="data_examples/openfold/records/P90561.json")

# Step 2 : Take the first residue and get it's atom indices
residue_index = 1
residue = structure_object.residues[residue_index]
residue_name = residue.name
print(f"Residue Name : {residue_name}")

# Step 3 : Create a tensor of zeros of shape (37,3) for positions and torch.eye of shape (8,4,4) for all the frames
# and tensor of zeros of shape 7,2
residue_atom_positions = torch.zeros((37, 3), device=device, dtype=dtype)
residue_frames = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(8, 1, 1)
residue_angles = torch.zeros((7, 2), device=device, dtype=dtype)

# Step 4 : Go through all atoms and set their coordinates
# To get their index use atom_types in constants.py
# Also create a dictionary called atom_position_dictionary containing for each atom the following
# key as the atom name
# value is a dictionary containing
# - global_coordinates : original x, y, z coordinates
# - frame : which will be the frame index that it is supposed to be in can be foudn in atom_frame_indices or rigid_group_atom_positions
# - frame_coordinates : equal to global_coordinates to begin with
# - current_frame_used : None (used to track how we express coordinates interatively, used for debugging)
atom_position_dictionary = {}
for i in range(residue.atom_count):
    atom = structure_object.atoms[residue.atom_start_index + i]
    atom_name = atom.name

    if atom_name in atom_to_index:
        atom_index = atom_to_index[atom_name]
        global_coordinates = torch.tensor(atom.experimental_coordinates, device=device, dtype=dtype)
        residue_atom_positions[atom_index] = global_coordinates

        # Get frame index for the specific amino acid type and atom type
        amino_acid_index = residue.amino_acid_index
        frame_index = int(atom_frame_indices[amino_acid_index, atom_index])

        atom_position_dictionary[atom_name] = {
            "global_coordinates": global_coordinates.clone(),
            "frame": frame_index,
            "frame_coordinates": global_coordinates.clone(),
            "current_frame_used": None
        }

# # print(residue_atom_positions.numpy())
# for k in range(residue.atom_count):
#     print(structure_object.atoms[residue.atom_start_index+k].name)
#     print(structure_object.atoms[residue.atom_start_index+k].experimental_coordinates)
# for k, v in atom_position_dictionary.items():
#     print(f"Key : {k}")
#     print(v["frame"])


# Step 5 : Set the coordinates of CA as the coordinates of the backbone translation
# Could get it from residue.center_atom_index or otherwise ?
# ex : vector CA->C
# ey : vector CA->N
# translation : CA position
# use these to create these three to create the transformation_backbone_frame using
# create_4x4_transform_matrix(ex,ey,translation)
# Test for me just see if global difference vector is equal to inframe vector ?
carbon_alpha_coordinates = atom_position_dictionary["CA"]["global_coordinates"]
carbon_coordinates = atom_position_dictionary["C"]["global_coordinates"]
nitrogen_coordinates = atom_position_dictionary["N"]["global_coordinates"]

vector_ca_to_c = carbon_coordinates - carbon_alpha_coordinates
vector_ca_to_n = nitrogen_coordinates - carbon_alpha_coordinates

transformation_backbone_frame = create_4x4_transform_matrix(ex=vector_ca_to_c,
                                                            ey=vector_ca_to_n,
                                                            translation_vector=carbon_alpha_coordinates)

# Test: Check if global difference vector is equal to in-frame vector
inverse_backbone_frame = invert_4x4_transform_matrix(transformation_backbone_frame)
# local_carbon_alpha_coordinates = apply_transformation_on_vector(inverse_backbone_frame, carbon_alpha_coordinates)
# local_carbon_coordinates = apply_transformation_on_vector(inverse_backbone_frame, carbon_coordinates)
# Set up the frame in residue frames
residue_frames[0] = transformation_backbone_frame
# print(residue_frames[0].numpy())

# Step 6 : Once this is done express all the atoms in this frame by left multiplying by the backbone inverse
# First invert the backbone transformation matrix using invert_4x4_transform_matrix
# Second use apply_transformation_on_vector of each of the positions
# The update is done on the atom_position_dictionary that we created and operates on the value frame_coordinates,
# so it is dynamically setting them to become their respective frame coordinates.
# update current_frame_used to 0 since we used the backbone frame
inverse_backbone_transformation_matrix = invert_4x4_transform_matrix(transformation_backbone_frame)

for atom_name, atom_data in atom_position_dictionary.items():
    current_global_coordinates = atom_data["frame_coordinates"]

    # Application of inverse of the backbone transform to get coordinates to the local frame of our residue.
    transformed_coordinates = apply_transformation_on_vector(
        transformation_matrix=inverse_backbone_transformation_matrix,
        vector=current_global_coordinates)

    atom_data["frame_coordinates"] = transformed_coordinates
    atom_data["current_frame_used"] = 0

# Step 7 : We move on to the phi frame (we will deal with the omega frame at the very end)
# ex : vector CA->N (using those in frame_coordinates)
# ey : vector CA->C (using those in frame_coordinates)
# translation : updated N position (using those in frame_coordinates)
# we are at frame index 2 (0 start indexing) so for the atoms that have the frame index
# update there frame coordinates using the inverse of the transformation matrix.
# update current_frame_used accordignly

# We use frame coordinates for the following steps
local_carbon_alpha = atom_position_dictionary["CA"]["frame_coordinates"]
local_nitrogen = atom_position_dictionary["N"]["frame_coordinates"]
local_carbon = atom_position_dictionary["C"]["frame_coordinates"]

vector_ca_to_n = local_nitrogen - local_carbon_alpha
vector_ca_to_c = local_carbon - local_carbon_alpha

# Compute the actual dihedral angle between C_{i-1}, N_i, CA_i, C_i
if residue.residue_index == 0:
    # Case where there is no previous residue, so set to (cos(0), sin(0))
    phi_angle = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
else:
    previous_residue = structure_object.residues[residue.residue_index - 1]

    previous_carbon_coordinates = None
    for i in range(previous_residue.atom_count):
        # Get the carbon coordinate from previous residue
        atom_in_prev = structure_object.atoms[previous_residue.atom_start_index + i]
        if atom_in_prev.name == "C":
            previous_carbon_coordinates = torch.tensor(atom_in_prev.experimental_coordinates, device=device,
                                                       dtype=dtype)
            break

    if previous_carbon_coordinates is not None:
        phi_angle = compute_dihedral_angle(
            point_1=previous_carbon_coordinates, point_2=nitrogen_coordinates,
            point_3=carbon_alpha_coordinates, point_4=carbon_coordinates)
    else:
        # This should not be triggered, so set to (cos(0), sin(0))
        print(f"There is a problem here for residue {residue.residue_index}")
        phi_angle = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

transformation_phi_frame = create_4x4_transform_matrix(ex=vector_ca_to_n,
                                                       ey=vector_ca_to_c,
                                                       translation_vector=local_nitrogen)

rotation_phi = make_transformation_matrix_around_ex(phi=phi_angle)
transformation_phi_frame = torch.matmul(transformation_phi_frame, rotation_phi)

# Set the residue angle and residue frame for phi
residue_angles[1] = phi_angle
residue_frames[2] = transformation_phi_frame

# Just for testing purposes
inverse_phi_transformation_matrix = invert_4x4_transform_matrix(transformation_phi_frame)
for atom_name, atom_data in atom_position_dictionary.items():
    if atom_data["frame"] == 2:
        current_coordinates = atom_data["frame_coordinates"]
        transformed_coordinates = apply_transformation_on_vector(
            transformation_matrix=inverse_phi_transformation_matrix,
            vector=current_coordinates)

        atom_data["frame_coordinates"] = transformed_coordinates
        atom_data["current_frame_used"] = 2

# Step 8 : We move on to the psi frame
# ex : vector CA->C (using those in frame_coordinates)
# ey : vector CA->N (using those in frame_coordinates)
# translation : updated C position (using those in frame_coordinates)
# Then get residue angle from two dihedral planes formed by (Ni,CAi,Ci) and (CAi,Ci,Oi)

# Setup local coordinates : todo to simplify
local_carbon_alpha_psi = atom_position_dictionary["CA"]["frame_coordinates"]
local_nitrogen_psi = atom_position_dictionary["N"]["frame_coordinates"]
local_carbon_psi = atom_position_dictionary["C"]["frame_coordinates"]

vector_ca_to_c_psi = local_carbon_psi - local_carbon_alpha_psi
vector_ca_to_n_psi = local_nitrogen_psi - local_carbon_alpha_psi

# Extract global coordinates for the dihedral calculation
oxygen_coordinates = atom_position_dictionary["O"]["global_coordinates"]

# AF2 explicitly uses N, CA, C, O of the SAME residue to compute the Psi angle
psi_angle = compute_dihedral_angle(point_1=nitrogen_coordinates, point_2=carbon_alpha_coordinates,
                                   point_3=carbon_coordinates, point_4=oxygen_coordinates)

rotation_psi = make_transformation_matrix_around_ex(phi=psi_angle)

base_psi_frame = create_4x4_transform_matrix(ex=vector_ca_to_c_psi,
                                             ey=vector_ca_to_n_psi,
                                             translation_vector=local_carbon_psi)
transformation_psi_frame = torch.matmul(base_psi_frame, rotation_psi)

# Set the residue angle and residue frame for psi
residue_angles[2] = psi_angle
residue_frames[3] = transformation_psi_frame

# Just for testing purposes
inverse_psi_transformation_matrix = invert_4x4_transform_matrix(transformation_psi_frame)
for atom_name, atom_data in atom_position_dictionary.items():
    if atom_data["frame"] == 3:
        current_coordinates = atom_data["frame_coordinates"]
        transformed_coordinates = apply_transformation_on_vector(
            transformation_matrix=inverse_psi_transformation_matrix,
            vector=current_coordinates
        )
        atom_data["frame_coordinates"] = transformed_coordinates
        atom_data["current_frame_used"] = 3

# print(residue_angles.numpy())
# print(transformation_phi_frame.numpy().round(2))
# frame_debugger(atom_position_dictionary, residue_name)

# Step 9 : We move on to the chi1 frame
# First check chi_angles_mask[amino_acid_index][0] if it is 0 then does not exist then no need to do anything just move on to the next step
# #SC0 can be found when looking at chi_angles_frame_centers[residue_xxx_name][0] if it does not exist then no need to do anything
# ex : vector CA->#SC0 (using those in frame_coordinates)
# ey : vector CA -> N (using those in frame_coordinates)
# translation : updated #SC0 position
# Todo we will have to add the #SC1 to a new variable to avoid looping to find it.
# To get the rotation compute dihedral angle of N -> CA -> #SCO -> #SC1

# we are at frame index 4 (0 start indexing) so for the atoms that have the frame index
# update there frame coordinates using the inverse of the transformation matrix.
# update current_frame_used accordignly

if chi_angles_mask[residue.amino_acid_index][0] == 0:
    residue_angles[3] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    # The frame was initialized as Identity in Step 3, which is correct for missing chi angles.
else:
    chi1_center_atom_0 = chi_dihedral_dictionary[residue_name]["atom_0"]
    chi1_center_atom_1 = chi_dihedral_dictionary[residue_name]["atom_1"]

    local_first_sidechain_atom_coordinates = atom_position_dictionary[chi1_center_atom_0]["frame_coordinates"]
    local_carbon_alpha_coordinates_for_chi1 = atom_position_dictionary["CA"]["frame_coordinates"]
    local_nitrogen_coordinates_for_chi1 = atom_position_dictionary["N"]["frame_coordinates"]

    vector_ca_to_first_sidechain = local_first_sidechain_atom_coordinates - local_carbon_alpha_coordinates_for_chi1
    vector_ca_to_nitrogen_for_chi1 = local_nitrogen_coordinates_for_chi1 - local_carbon_alpha_coordinates_for_chi1

    transformation_chi1_base_frame = create_4x4_transform_matrix(
        ex=vector_ca_to_first_sidechain,
        ey=vector_ca_to_nitrogen_for_chi1,
        translation_vector=local_first_sidechain_atom_coordinates)

    global_nitrogen_coordinates = atom_position_dictionary["N"]["global_coordinates"]
    global_carbon_alpha_coordinates = atom_position_dictionary["CA"]["global_coordinates"]
    global_first_sidechain_atom_coordinates = atom_position_dictionary[chi1_center_atom_0]["global_coordinates"]
    global_second_sidechain_atom_coordinates = atom_position_dictionary[chi1_center_atom_1]["global_coordinates"]

    chi1_dihedral_angle = compute_dihedral_angle(
        point_1=global_nitrogen_coordinates,
        point_2=global_carbon_alpha_coordinates,
        point_3=global_first_sidechain_atom_coordinates,
        point_4=global_second_sidechain_atom_coordinates)

    rotation_matrix_chi1 = make_transformation_matrix_around_ex(phi=chi1_dihedral_angle)
    transformation_chi1_frame = torch.matmul(transformation_chi1_base_frame, rotation_matrix_chi1)

    # Set chi1 angle
    residue_angles[3] = chi1_dihedral_angle
    residue_frames[4] = transformation_chi1_frame

    # Set the residue angle and residue frame for chi1
    inverse_chi1_transformation_matrix = invert_4x4_transform_matrix(transformation_chi1_frame)

    # Just for testing purposes
    for atom_name, atom_data in atom_position_dictionary.items():
        if atom_data["frame"] == 4:
            current_coordinates = atom_data["frame_coordinates"]
            transformed_coordinates = apply_transformation_on_vector(
                transformation_matrix=inverse_chi1_transformation_matrix, vector=current_coordinates)

            atom_data["frame_coordinates"] = transformed_coordinates
            atom_data["current_frame_used"] = 4

frame_debugger(atom_position_dictionary, residue_name, frame_to_consider=4)
# Step 10 : We move on to the chi2 frame and do the same thing
# ex : #SC0 -> #SC1 (using those in frame_coordinates)
# ey : #SC0 -> CA (using those in frame_coordinates)
# translation :  #SC1 (using those in frame_coordinates)
# update residue_angles, frame_coordinates and current_frame_used accordignly


# Step 11 : We move on to the chi3 frame and do the same thing
# ex : #SC1 -> #SC2 (using those in frame_coordinates)
# ey : #SC1 -> #SC0 (using those in frame_coordinates)
# translation : #SC2 (using those in frame_coordinates)
# update residue_angles, frame_coordinates and current_frame_used accordignly

# Step 12 : We move on to the chi4 frame and do the same thing
# ex : #SC2 -> #SC3 (using those in frame_coordinates)
# ey : #SC2 -> #SC1 (using those in frame_coordinates)
# translation : #SC3 (using those in frame_coordinates)
# update residue_angles, frame_coordinates and current_frame_used accordignly

# Last step is where we deal with the omega
# ex : C -> N (nitrogen of the next residue if it exist, using those in global coordinates)
# ey : CA -> C (using those in global coordinates since we will use N of next residue)
# translation : C (using global coordinates)
# create_4x4_transform_matrix(ex,ey,translation)
# if N (next) it does not exist
# set transformation matrix to identity matrix with translation equals to (C but frame coordinates)
# set the residue_angle accodingly, if N did not exist just 0 degree angle.


exit()
batch_size = 2
number_residues = 5

pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)

for k in range(number_residues):
    pred_tr_m[k, :3, -1] += k + 1
    gt_tr_m[k, :3, -1] += k + 2

pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))

pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
    (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)

gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
    (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,
                                                                                                        1)

print(pred_tr_m.shape)
print(gt_tr_m.shape)
exit()
fape_loss = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                              predicted_positions=pred_carbon_alpha_positions,
                              ground_truth_transformation_matrix=gt_tr_m,
                              ground_truth_positions=gt_carbon_alpha_positions,
                              length_scaler=2,
                              epsilon=2e-4,
                              distance_clamp=2.0)

exit()


def create_all_atom_positions(batch_size: int, number_residues: int, flip: bool = False, random=False):
    if random:
        positions = torch.randperm(number_residues * 37 * 3).reshape(number_residues, 37, 3)
    else:
        positions = torch.arange(number_residues * 37 * 3).reshape(number_residues, 37, 3)

    if flip:
        positions = torch.flip(positions, dims=[-2])

    positions = positions.to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return positions


batch_size = 8
number_residues = 10
dtype = torch.float32
scale = 30
threshold = 15
distance_thresholds = [9.5, 1.0, 2.0, 4.0]

# Get the residues of the sequence
amino_acid_residues = (torch.arange(batch_size * number_residues) % 20).reshape(batch_size, number_residues)
amino_acid_residues = torch.flip(amino_acid_residues, dims=[-1])

# Create the positions of predictions and ground truths
# (batch, number_residues, 37, 3)
prediction_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues,
                                                 random=True) / scale
ground_truth_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues,
                                                   flip=True) / (scale / 2)

local_difference_distance_test = compute_local_distance_difference_test(
    # prediction_positions=ground_truth_positions,
    prediction_positions=prediction_positions,
    ground_truth_positions=ground_truth_positions,
    distance_thresholds=distance_thresholds)

from architecture_modules.lddt_module.lddt_module import LddtModule

# Create single_representation
embedding_dimension = 100
single_representation = torch.randperm(batch_size * number_residues * embedding_dimension, dtype=torch.float64).reshape(
    batch_size,
    number_residues,
    embedding_dimension)

lddt_module = LddtModule(single_representation_embedding=embedding_dimension,
                         intermediate_embedding=int(embedding_dimension / 4),
                         device=torch.device("cpu"),
                         dtype=torch.float64)

lddt_logits, predicted_lddt_probabilities, plddt = lddt_module(single_representation=single_representation)

plddt_loss = compute_plddt_loss(
    ground_truth_lddt=local_difference_distance_test,
    predicted_lddt_logits=lddt_logits,
    lddt_bins=lddt_module.lddt_bins
)
# def create_transformation_matrices(batch_size: int,
#                                    number_residues: int,
#                                    number_frames: int = 8,
#                                    delta=2):
#     transformation_matrix = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, number_frames, 1, 1)
#
#     # Replace the last column (equivalent to the translation with some known value > simple (x,x,x) translation)
#     for k in range(number_residues):
#         for j in range(number_frames):
#             transformation_matrix[k, j, :3, -1] += k + j + 1 + delta
#
#     # Accommodate batch
#     transformation_matrix = transformation_matrix.repeat(batch_size, 1, 1, 1, 1)
#
#     return transformation_matrix
#
#
# # The position matrices
# ground_truth_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues, flip=False)
# predicted_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues, flip=True)
# # Shape (number_residues, 37)
# alternative_positions = alternative_position_mask[amino_acid_residues].unsqueeze(dim=-1).repeat(1, 1, 1, 3)
# alternative_ground_truth_positions = torch.gather(ground_truth_positions, dim=2, index=alternative_positions)
#
# # The transformation matrices
# ground_truth_transformation_matrix = create_transformation_matrices(batch_size=batch_size,
#                                                                     number_residues=number_residues,
#                                                                     number_frames=number_frames, delta=0)
# predicted_transformation_matrix = create_transformation_matrices(batch_size=batch_size,
#                                                                  number_residues=number_residues,
#                                                                  number_frames=number_frames, delta=1)
#
# # Swap change the rotations of the transformation matrices
# alternative_rotations = alternative_angle_mask[amino_acid_residues]
# alternative_rotations = alternative_rotations.repeat(batch_size, 1, 1, 1)
# residue_angles = torch.tensor([1.0, 0.0]).repeat(batch_size, number_residues, 7, 1)
# alternative_ground_truth_transformation_matrix = create_alternative_truth_transformation_matrix(
#     transformation_matrix=ground_truth_transformation_matrix,
#     sequence_amino_acid_labels=amino_acid_residues)
#
#
# def rename_symetric_ground_truth_metrics(predicted_transformation_matrix,
#                                          predicted_positions,
#                                          ground_truth_transformation_matrix,
#                                          ground_truth_positions,
#                                          alternative_ground_truth_transformation_matrix,
#                                          alternative_ground_truth_positions,
#                                          sequence_amino_acid_labels):
#     # Important : We assume that there is no batch in our inputs
#
#     # Get tensors that will be returned
#     modified_ground_truth_positions = ground_truth_positions.clone()
#     modified_ground_truth_transformation_matrix = ground_truth_transformation_matrix.clone()
#
#     # Get non-ambiguous positions
#     sequence_ambiguous_positions_masks = ambiguous_position_mask[sequence_amino_acid_labels]
#     sequence_non_ambiguous_position_masks = ~sequence_ambiguous_positions_masks
#
#     # Gets all the non ambigouous positions : (non_ambiguous_atoms_of_sequence, 3)
#     sequence_unambiguous_predicted_positions = predicted_positions[sequence_non_ambiguous_position_masks]
#     sequence_unambiguous_ground_truth_positions = ground_truth_positions[sequence_non_ambiguous_position_masks]
#
#     # Go through all residues and only evaluate the amino acid residues with ambigous atoms
#     for index, residue_index in enumerate(sequence_amino_acid_labels):
#
#         # Skip if this residue has no ambiguous atoms.
#         if not sequence_ambiguous_positions_masks[index].any():
#             continue
#
#         # Get current residue positions
#         pred_res_pos = predicted_positions[index]
#         gt_res_pos = ground_truth_positions[index]
#         alt_gt_res_pos = alternative_ground_truth_positions[index]
#
#         # Get current residue ambiguous positions
#         pred_ambiguous_positions = pred_res_pos[ambiguous_position_mask[residue_index]]
#         gt_ambiguous_positions = gt_res_pos[ambiguous_position_mask[residue_index]]
#         alt_gt_ambiguous_positions = alt_gt_res_pos[ambiguous_position_mask[residue_index]]
#
#         # Get the different distances
#         # - predicitions<->predictions
#         distance_predictions = torch.cdist(x1=pred_ambiguous_positions,
#                                            x2=sequence_unambiguous_predicted_positions)
#
#         # - ground_truth<->ground_truth
#         distance_ground_truths = torch.cdist(x1=gt_ambiguous_positions,
#                                              x2=sequence_unambiguous_ground_truth_positions)
#
#         # - alternative_ground_truth <-> ground_truth
#         distance_alternative_ground_truths = torch.cdist(x1=alt_gt_ambiguous_positions,
#                                                          x2=sequence_unambiguous_ground_truth_positions)
#
#         # Left element abs(predictions-alt_ground_truth)
#         left_side = torch.sum(torch.abs(distance_predictions - distance_alternative_ground_truths))
#
#         # Right element abs(predictions-ground_truth)
#         right_side = torch.sum(torch.abs(distance_predictions - distance_ground_truths))
#
#         if left_side < right_side:
#             modified_ground_truth_positions[index] = alternative_ground_truth_positions[index]
#             modified_ground_truth_transformation_matrix[index] = alternative_ground_truth_transformation_matrix[index]
#
#     return modified_ground_truth_positions, modified_ground_truth_transformation_matrix
#
#
# for batch_index in range(batch_size):
#     modified_ground_truth_positions, modified_ground_truth_transformation_matrix = rename_symetric_ground_truth_metrics(
#         predicted_transformation_matrix=predicted_transformation_matrix[batch_index],
#         predicted_positions=predicted_positions[batch_index],
#         ground_truth_transformation_matrix=ground_truth_transformation_matrix[batch_index],
#         ground_truth_positions=ground_truth_positions[batch_index],
#         alternative_ground_truth_transformation_matrix=alternative_ground_truth_transformation_matrix[batch_index],
#         alternative_ground_truth_positions=alternative_ground_truth_positions[batch_index],
#         sequence_amino_acid_labels=amino_acid_residues[batch_index],
#     )

# for b in range(batch_size):
#     for r in range(number_residues):
#         aa_r = r % 20
#         if index_to_xxx[aa_r] not in ["ASP", "GLU", "TYR", "PHE"]:
#             continue
#         for n in range(37):
#             gt_atom = ground_truth_positions[b, aa_r, n].numpy()
#             alt_atom = alternative_ground_truth_positions[b, aa_r, n].numpy()
#
#             if sum(alt_atom - gt_atom) != 0:
#                 print(f"Amino Acid Residue {index_to_xxx[aa_r]} For Atom {n:02}")
#                 print(15 * '-')
#                 print(gt_atom, alt_atom)
#                 print(30 * '-')
# exit()
# print(residue_angles)
# print_tensor_shape(tensor=alternative_rotations)
# print_tensor_shape(tensor=residue_angles)
# residue_angles = residue_angles*alternative_rotations
# print(residue_angles)
# print(alternative_rotations)


# for b in range(batch_size):
#     for r in range(number_residues):
#         aa_r = r % 20
#         for f in range(number_frames):
#             print(f"Amino Acid Residue {index_to_xxx[aa_r]} For Angle Frame {f:02}")
#             print(ground_truth_transformation_matrix[b, aa_r, f].numpy())
#             print(15 * '-')
#             print(alternative_ground_truth_transformation_matrix[b, aa_r, f].numpy())
#             print(30 * '-')
# exit()
# alternative_ground_truth_transformation_matrix =


# scaler = 3
# pred_gt_delta = 2
# # Predicted angles number_residues, 7, 2
# predicted_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
# predicted_y = (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
# predicted_unnormalised_torsion_angles = torch.stack(tensors=[predicted_x, predicted_y], dim=-1)
#
# # Ground Truth angles number_residues, 7, 2
# gt_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
# gt_y = pred_gt_delta * (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
# ground_truth_torsion_angles = torch.nn.functional.normalize(torch.stack(tensors=[gt_x, gt_y], dim=-1), dim=-1)
#
#
#

#
#
# # Get the alternative truths for the torsion angles
# alternative_scaler = alternative_angle_mask[amino_acid_residues]
# # print(f"{angle_alternative_truth_mask.shape=}")
# # print(f"{residues_to_change.shape=}")
# # print(alternative_scaler)
# alternative_ground_truth_torsion_angles = alternative_scaler * ground_truth_torsion_angles
# # print(ground_truth_torsion_angles.numpy())
# # print(20*'--')
# # print(alternative_ground_truth_torsion_angles.numpy())
#
# # Add batch size to the different tensors
# predicted_unnormalised_torsion_angles = predicted_unnormalised_torsion_angles.repeat(batch_size, 1, 1, 1)
# ground_truth_torsion_angles = ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)
# alternative_ground_truth_torsion_angles = alternative_ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)
#
#
# auxillary_loss = compute_torsion_angle_loss(
#     predicted_unnormalised_angles=predicted_unnormalised_torsion_angles,
#     ground_truth_angles=ground_truth_torsion_angles,
#     alternative_ground_truth_angles=alternative_ground_truth_torsion_angles,
#     angle_norm_loss_scaler=0.2)
# print(auxillary_loss.numpy())
# # pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
# # gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
# #
# # for k in range(number_residues):
# #     pred_tr_m[k, :3, -1] += k + 1
# #     gt_tr_m[k, :3, -1] += k + 2
# #
# # pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
# # gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))
# #
# # pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
# #     (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)
# #
# # gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
# #     (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,1)
