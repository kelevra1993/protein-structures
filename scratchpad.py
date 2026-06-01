import torch
import numpy as np

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list
from utilities.loss_utilities import compute_fape_loss
from utilities.constants import alternative_angle_mask

batch_size = 2
number_residues = 10
scaler = 3
pred_gt_delta = 1
dtype = torch.float32

# Predicted angles number_residues, 7, 2
predicted_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
predicted_y = (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
predicted_unnormalised_torsion_angles = torch.stack(tensors=[predicted_x, predicted_y], dim=-1)

# Ground Truth angles number_residues, 7, 2
gt_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
gt_y = pred_gt_delta * (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
ground_truth_torsion_angles = torch.nn.functional.normalize(torch.stack(tensors=[gt_x, gt_y], dim=-1), dim=-1)

# print(ground_truth_torsion_angles)
# print(torch.linalg.norm(ground_truth_torsion_angles, dim=-1))

# Get the residues of the sequence
amino_acid_residues = torch.arange(number_residues) % 20
# amino_acid_residues = torch.flip(amino_acid_residues, dims=[0])


# Get the alternative truths for the torsion angles
alternative_scaler = alternative_angle_mask[amino_acid_residues]
# print(f"{angle_alternative_truth_mask.shape=}")
# print(f"{residues_to_change.shape=}")
# print(alternative_scaler)
alternative_ground_truth_torsion_angles = alternative_scaler * ground_truth_torsion_angles
print(alternative_ground_truth_torsion_angles)


def compute_torsion_angle_loss(predicted_unnormalised_angles,
                               ground_truth_angles,
                               alternative_ground_truth_angles,
                               angle_norm_loss_scaler):
    norm_predicted_angles = torch.linalg.norm(predicted_unnormalised_angles, dim=-1)
    print_tensor_shape(tensor=norm_predicted_angles, name="norm_predicted_angles")

    # TODO Make sure that prediction angles is indeed normalised
    prediction_angles = torch.nn.functional.normalize(predicted_unnormalised_torsion_angles, dim=-1)
    print_tensor_shape(tensor=prediction_angles, name="prediction_angles")

    #
    pred_gt_difference = torch.linalg.norm(prediction_angles - ground_truth_angles) ** 2
    print_tensor_shape(tensor=pred_gt_difference, name="pred_gt_difference")
    pred_alternative_gt_difference = torch.linalg.norm(prediction_angles - alternative_ground_truth_angles) ** 2
    print_tensor_shape(tensor=pred_gt_difference, name="pred_gt_difference")

    torsion_loss = torch.mean(torch.minimum(input=pred_gt_difference, other=pred_alternative_gt_difference),
                              dim=[-2, -1])
    print_tensor_shape(tensor=torsion_loss, name="torsion_loss")
    print(f"Torsion Loss = {torsion_loss.item()}")

    # TODO Test that this is does indeed return what is expected, therefore never a negative number.
    angle_norm_loss = torch.mean(torch.abs(norm_predicted_angles - 1))

    return torsion_loss + angle_norm_loss_scaler * angle_norm_loss

# pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
# gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
#
# for k in range(number_residues):
#     pred_tr_m[k, :3, -1] += k + 1
#     gt_tr_m[k, :3, -1] += k + 2
#
# pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
# gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))
#
# pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
#     (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)
#
# gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
#     (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,1)
