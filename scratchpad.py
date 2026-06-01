import torch
import numpy as np

np.set_printoptions(linewidth=200, threshold=np.inf)

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss
from utilities.constants import alternative_angle_mask

batch_size = 2
number_residues = 4
scaler = 3
pred_gt_delta = 2
dtype = torch.float32

# Predicted angles number_residues, 7, 2
predicted_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
predicted_y = (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
predicted_unnormalised_torsion_angles = torch.stack(tensors=[predicted_x, predicted_y], dim=-1)

# Ground Truth angles number_residues, 7, 2
gt_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
gt_y = pred_gt_delta * (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
ground_truth_torsion_angles = torch.nn.functional.normalize(torch.stack(tensors=[gt_x, gt_y], dim=-1), dim=-1)



# Get the residues of the sequence
amino_acid_residues = torch.arange(number_residues) % 20
# amino_acid_residues = torch.flip(amino_acid_residues, dims=[0])


# Get the alternative truths for the torsion angles
alternative_scaler = alternative_angle_mask[amino_acid_residues]
# print(f"{angle_alternative_truth_mask.shape=}")
# print(f"{residues_to_change.shape=}")
# print(alternative_scaler)
alternative_ground_truth_torsion_angles = alternative_scaler * ground_truth_torsion_angles
# print(ground_truth_torsion_angles.numpy())
# print(20*'--')
# print(alternative_ground_truth_torsion_angles.numpy())

# Add batch size to the different tensors
predicted_unnormalised_torsion_angles = predicted_unnormalised_torsion_angles.repeat(batch_size, 1, 1, 1)
ground_truth_torsion_angles = ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)
alternative_ground_truth_torsion_angles = alternative_ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)


auxillary_loss = compute_torsion_angle_loss(
    predicted_unnormalised_angles=predicted_unnormalised_torsion_angles,
    ground_truth_angles=ground_truth_torsion_angles,
    alternative_ground_truth_angles=alternative_ground_truth_torsion_angles,
    angle_norm_loss_scaler=0.2)
print(auxillary_loss.numpy())
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
