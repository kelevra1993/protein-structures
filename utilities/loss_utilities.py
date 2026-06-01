import torch
from utilities.geometry_utilities import invert_4x4_transform_matrix, apply_transformation_on_vector


def compute_fape_loss(predicted_transformation_matrix: torch.Tensor, predicted_positions: torch.Tensor,
                      ground_truth_transformation_matrix: torch.Tensor, ground_truth_positions: torch.Tensor,
                      length_scaler: int = 10, epsilon: float = 1e-4, distance_clamp: float = 10.0):
    """
    Computes the Frame Aligned Point Error (FAPE) loss between predicted and ground truth structures.

    FAPE is a critical loss function in AlphaFold II, used to measure structural discrepancy by aligning
    predicted and ground truth structures through their local reference frames. For each pair of residues
    (i, j), residue j's position is transformed into the local coordinate system of residue i's frame.
    The L2 distance between these transformed positions is then calculated and averaged across all pairs.
    This approach ensures the loss is invariant to global rotations and translations and captures
    relative structural relationships.

    Args:
        predicted_transformation_matrix: Predicted local transformation frames for each residue.
            Expected shape: (*batch_dims, number_residues, 4, 4)
        predicted_positions: Predicted 3D Cartesian coordinates (typically for CA atoms).
            Expected shape: (*batch_dims, number_residues, 3)
        ground_truth_transformation_matrix: Ground truth local transformation frames.
            Expected shape: (*batch_dims, number_residues, 4, 4)
        ground_truth_positions: Ground truth 3D Cartesian coordinates.
            Expected shape: (*batch_dims, number_residues, 3)
        length_scaler: A normalization constant, typically 10.0 Angstroms.
        epsilon: Small constant for numerical stability when calculating the square root of distances.
        distance_clamp: A cutoff distance (in Angstroms) above which the error is clamped.

    Returns:
        fape_loss: The computed FAPE loss value for each batch.
            Shape: (*batch_dims)
    """
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

    # Apply inverted transformation on both predictions and on the ground truth values
    transformed_predictions = apply_transformation_on_vector(
        transformation_matrix=predicted_inverted_transformation_matrix,
        vector=predicted_positions)

    transformed_ground_truths = apply_transformation_on_vector(
        transformation_matrix=ground_truth_inverted_transformation_matrix,
        vector=ground_truth_positions)

    # Compute distance (while adding a small epsilon to avoid derivatives at 0)
    # Clamp the distance accordignly and compute the mean on the width and height.
    squared_norm_tensor = torch.linalg.vector_norm((transformed_predictions - transformed_ground_truths), dim=-1) ** 2
    distance_matrix = torch.clamp(torch.sqrt(squared_norm_tensor + epsilon), max=distance_clamp)

    fape_loss = torch.mean(distance_matrix, dim=[-2, -1]) / length_scaler

    return fape_loss


def compute_torsion_angle_loss(predicted_unnormalised_angles: torch.Tensor,
                               ground_truth_angles: torch.Tensor,
                               alternative_ground_truth_angles: torch.Tensor,
                               angle_norm_loss_scaler: float = 0.02) -> torch.Tensor:
    """
    Computes the torsion angle loss and the angle unit norm loss.

    This loss function is used in AlphaFold II to supervise the prediction of backbone (omega, phi, psi)
    and side-chain (chi1, chi2, chi3, chi4) torsion angles. Torsion angles are represented as points
    (cos(theta), sin(theta)) on the unit circle. The loss consists of two components:

    1. Torsion Loss: The squared L2 distance between predicted and ground truth angles. It accounts for
       side-chain symmetries by taking the minimum distance between the prediction and two possible
       ground truth configurations (standard and alternative).
    2. Angle Norm Loss: Penalizes the model when the predicted (unnormalized) (cos, sin) pairs do not
       lie on the unit circle, ensuring they can be interpreted as valid rotation angles.

    Args:
        predicted_unnormalised_angles: Unnormalized predicted torsion angles.
            Expected shape: (*batch_dims, number_residues, 7, 2)
        ground_truth_angles: Ground truth torsion angles.
            Expected shape: (*batch_dims, number_residues, 7, 2)
        alternative_ground_truth_angles: Alternative ground truth torsion angles,
            accounting for 180-degree symmetries in certain side chains (e.g., TYR chi2).
            Expected shape: (*batch_dims, number_residues, 7, 2)
        angle_norm_loss_scaler: Scaling factor for the angle unit norm loss component.

    Returns:
        total_torsion_loss: Combined torsion and normalization loss.
            Shape: (*batch_dims)
    """

    # Get prediction angle norms used for angle unit norm loss (batch_size, number_residues, 7)
    norm_predicted_angles = torch.linalg.norm(predicted_unnormalised_angles, dim=-1)

    # Compute prediction angles (cos(phi), sin(phi)) (batch_size, number_residues, 7, 2)
    prediction_angles = torch.nn.functional.normalize(predicted_unnormalised_angles, dim=-1)

    # Compute squared norm difference for both ground truth and it's alternative to the predicted angles.
    pred_gt_difference = torch.linalg.norm((prediction_angles - ground_truth_angles), dim=-1) ** 2
    pred_alternative_gt_difference = torch.linalg.norm((prediction_angles - alternative_ground_truth_angles),
                                                       dim=-1) ** 2

    # Compute torsion loss (*batch_dims)
    torsion_loss = torch.mean(torch.minimum(input=pred_gt_difference, other=pred_alternative_gt_difference),
                              dim=[-2, -1])

    # Todo : Later try to keep track of these norms to see if they tend to go towards 1
    # Be careful, we need the absolute value here to target a positive norm
    # Compute angle normalisation loss (*batch_dims)
    angle_norm_loss = torch.mean(torch.abs(norm_predicted_angles - 1), dim=[-2, -1])

    return torsion_loss + angle_norm_loss_scaler * angle_norm_loss
