import torch
import numpy as np

np.set_printoptions(linewidth=200, threshold=np.inf)

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss
from utilities.constants import alternative_angle_mask, alternative_position_mask, index_to_xxx, \
    ambiguous_position_mask, atom_types
from utilities.geometry_utilities import create_alternative_truth_transformation_matrix


# todo to be used to create positions
def create_all_atom_positions(batch_size: int, number_residues: int, flip: bool = False, random=False):
    if random:
        positions = torch.randperm(number_residues * 37 * 3).reshape(number_residues, 37, 3)
    else:
        positions = torch.arange(number_residues * 37 * 3).reshape(number_residues, 37, 3)

    if flip:
        positions = torch.flip(positions, dims=[-2])

    positions = positions.to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return positions


def create_transformation_matrices(batch_size: int,
                                   number_residues: int,
                                   number_frames: int = 8,
                                   delta=2):
    transformation_matrix = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, number_frames, 1, 1)

    # Replace the last column (equivalent to the translation with some known value > simple (x,x,x) translation)
    for k in range(number_residues):
        for j in range(number_frames):
            transformation_matrix[k, j, :3, -1] += k + j + 1 + delta

    # Accommodate batch
    transformation_matrix = transformation_matrix.repeat(batch_size, 1, 1, 1, 1)

    return transformation_matrix


batch_size = 8
number_residues = 5
number_frames = 8
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


def compute_local_distance_difference_test(prediction_positions,
                                           ground_truth_positions,
                                           clamp_threshold=15.0,
                                           distance_thresholds=None):
    batch_size, number_residues = prediction_positions.shape[:2]
    device = prediction_positions.device
    dtype = prediction_positions.dtype

    if not distance_thresholds:
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]

    # First Extract the carbon alpha postion
    carbon_alpha_index = atom_types.index("CA")

    # Alpha Carbon Positions
    predictions_ca_positions = prediction_positions[..., carbon_alpha_index, :]
    ground_truth_ca_positions = ground_truth_positions[..., carbon_alpha_index, :]

    # Compute distances
    prediction_distances = torch.cdist(predictions_ca_positions, predictions_ca_positions)
    ground_truth_distances = torch.cdist(ground_truth_ca_positions, ground_truth_ca_positions)

    # Boolean identity matrix of shape (number_residues, number_residues) used to avoid diagonals.
    diagonal_mask = torch.eye(number_residues, dtype=torch.bool, device=ground_truth_distances.device)
    off_diagonal_mask = ~diagonal_mask

    # Get a mask for pair distances that are not in the diagonal (avoid self difference)
    # and below the clamp_threshold
    considered_ca_pairs = torch.bitwise_and(input=off_diagonal_mask,
                                            other=(ground_truth_distances < clamp_threshold))

    # Get number of pairs to consider for each residue
    considered_ca_pair_counts = torch.sum(considered_ca_pairs.to(torch.float64), dim=-1)

    # Difference Distance Predictions and Ground Truth
    # Shape -> (batch_size, number_residues, nmber_residues)
    difference_prediction_ground_truth_distance = torch.abs(prediction_distances - ground_truth_distances)

    # Used to scale the final lddt matrix
    L = len(distance_thresholds)

    # Since we are prediciting the per residue local distance difference test
    # we sum over the residue columns thus the shape (batch_size, number_residues)
    local_difference_distance_test = torch.zeros((batch_size, number_residues), device=device, dtype=dtype)

    # Here be very careful, we set the pairs that should not be considered (~considered) to -1
    difference_prediction_ground_truth_distance[~considered_ca_pairs] = -1

    for current_threshold in distance_thresholds:
        # Now get indices of distance pairs >0 and under the current threshold value
        current_indices = torch.bitwise_and(
            input=(difference_prediction_ground_truth_distance >= 0),
            other=(difference_prediction_ground_truth_distance < current_threshold),
        )

        # Count the number of correct predictions and add them to the lddt matrix
        number_accurate_distances = torch.sum(current_indices.to(torch.float64), dim=-1)
        local_difference_distance_test += number_accurate_distances

    # Normalise the local difference distance test
    local_difference_distance_test /= (L * considered_ca_pair_counts)

    return local_difference_distance_test


local_difference_distance_test = compute_local_distance_difference_test(prediction_positions=prediction_positions,
                                                                        ground_truth_positions=ground_truth_positions,
                                                                        distance_thresholds=distance_thresholds)
print(local_difference_distance_test)
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
