import torch
import numpy as np

from tests.full_model.test_model import batch_size
from utilities.tensor_utilities import print_tensor_shape
from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector


# todo to review the epsilon
# todo this is just the implementation for Carbon alphas
def compute_fape_loss(predicted_transformation_matrix: torch.Tensor, predicted_positions: torch.Tensor,
                      ground_truth_transformation_matrix: torch.Tensor, ground_truth_positions: torch.Tensor,
                      length_scaler: int = 1, epsilon: float = 1e-4):
    batch_shape = predicted_transformation_matrix.shape[:-3]
    number_residues = predicted_transformation_matrix.shape[-3]

    # Expanded Shapes
    tr_expanded_shape = (batch_shape + (number_residues, number_residues, 4, 4))
    ps_expanded_shape = (batch_shape + (number_residues, number_residues, 3))

    # Invert Transformation matrices, unsqueeze them and broadcast them to be prepared for application of positions
    predicted_inverted_transformation_matrix = invert_4x4_transform_matrix(
        transformation_matrix=predicted_transformation_matrix).unsqueeze(dim=-3).broadcast_to(tr_expanded_shape)
    ground_truth_inverted_transformation_matrix = invert_4x4_transform_matrix(
        transformation_matrix=ground_truth_transformation_matrix).unsqueeze(dim=-3).broadcast_to(tr_expanded_shape)

    # Prepare positions before application
    predicted_positions = predicted_positions.unsqueeze(dim=-3).broadcast_to(ps_expanded_shape)
    ground_truth_positions = ground_truth_positions.unsqueeze(dim=-3).broadcast_to(ps_expanded_shape)

    transformed_predictions = apply_transformation_on_vector(
        transformation_matrix=predicted_inverted_transformation_matrix,
        vector=predicted_positions)

    transformed_ground_truths = apply_transformation_on_vector(
        transformation_matrix=ground_truth_inverted_transformation_matrix,
        vector=ground_truth_positions)

    print(f"{list(predicted_inverted_transformation_matrix.shape)=}")
    print(f"{list(predicted_positions.shape)=}")
    print(f"{list(transformed_predictions.shape)=}")

    return transformed_predictions, transformed_ground_truths


batch_size = 1
number_residues = 2

pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)

for k in range(number_residues):
    pred_tr_m[k, :3, -1] += k + 1
    gt_tr_m[k, :3, -1] += k + 2

pred_tr_m = pred_tr_m.broadcast_to((batch_size, number_residues, 4, 4))

carbon_alpha_positions = torch.ones(number_residues, 3) * (
    torch.arange(1, number_residues + 1, 1).unsqueeze(dim=-1)).to(torch.float64)

tr_pred, tr_gt = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                                   predicted_positions=carbon_alpha_positions,
                                   ground_truth_transformation_matrix=gt_tr_m,
                                   ground_truth_positions=carbon_alpha_positions,
                                   length_scaler=1, epsilon=1e-4)

for b in range(batch_size):
    for k in range(number_residues):
        print(90 * '-')
        print(f'Prediction : {tr_pred[b][number_residues-1][k]=}')
        print(f'Ground Truth : {tr_gt[b][number_residues-1][k]=}')
        print(90*'-')