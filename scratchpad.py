import torch
import numpy as np

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list
from utilities.loss_utilities import compute_fape_loss

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
    (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,1)

fape_loss = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                              predicted_positions=pred_carbon_alpha_positions,
                              ground_truth_transformation_matrix=gt_tr_m,
                              ground_truth_positions=gt_carbon_alpha_positions,
                              length_scaler=2,
                              epsilon=2e-4,
                              distance_clamp=2.0)
