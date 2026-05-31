import torch
from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector


def compute_fape_loss(predicted_transformation_matrix: torch.Tensor, predicted_positions: torch.Tensor,
                      ground_truth_transformation_matrix: torch.Tensor, ground_truth_positions: torch.Tensor,
                      length_scaler: int = 10, epsilon: float = 1e-4, distance_clamp: float = 10.0):
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

    squared_norm_tensor = torch.linalg.vector_norm((transformed_predictions - transformed_ground_truths), dim=-1) ** 2
    distance_matrix = torch.clamp(torch.sqrt(squared_norm_tensor + epsilon), max=distance_clamp)

    fape_loss = torch.mean(distance_matrix, dim=[-2, -1]) / length_scaler

    return fape_loss
