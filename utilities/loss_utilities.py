import torch

from utilities.constants import ambiguous_position_mask, atom_types
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


def compute_distogram_loss(distogram_logits: torch.Tensor, distogram_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the distogram loss using cross-entropy between predicted logits and ground truth labels.

    The distogram loss supervises the pair representation by comparing the predicted distance 
    distributions between all residue pairs against the true discretized distances.

    Args:
        distogram_logits: Unnormalized predicted distance bin logits.
            Expected shape: (*batch_dims, number_residues, number_residues, 64)
        distogram_labels: Ground truth distance bin indices.
            Expected shape: (*batch_dims, number_residues, number_residues)

    Returns:
        distogram_loss: The mean cross-entropy loss.
            Shape: ()
    """
    # Move dimensions of distogram predictions so that the class prediction is at dimension 1 :
    # (Batch, classes chanel, d1, d2, ...) as CrossEntropyLoss expects it
    # therefore move the 64 bins (dim -1) to index 1.
    logits = distogram_logits.movedim(source=-1, destination=1)

    # Compute cross entropy loss. Reduction is "mean" as per project standard.
    distogram_loss = torch.nn.functional.cross_entropy(input=logits, target=distogram_labels, reduction="mean")

    return distogram_loss


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

    # Compute torsion loss (batch_size, 1)
    torsion_loss = torch.mean(torch.minimum(input=pred_gt_difference, other=pred_alternative_gt_difference),
                              dim=[-2, -1])

    # Todo : Later try to keep track of these norms to see if they tend to go towards 1
    # Be careful, we need the absolute value here to target a positive norm
    # Compute angle normalisation loss (batch_size, 1)
    angle_norm_loss = torch.mean(torch.abs(norm_predicted_angles - 1), dim=[-2, -1])

    return torsion_loss + angle_norm_loss_scaler * angle_norm_loss


def rename_symetric_ground_truth_metrics(predicted_positions: torch.Tensor,
                                         ground_truth_transformation_matrix: torch.Tensor,
                                         ground_truth_positions: torch.Tensor,
                                         alternative_ground_truth_transformation_matrix: torch.Tensor,
                                         alternative_ground_truth_positions: torch.Tensor,
                                         sequence_amino_acid_labels: torch.Tensor):
    """
    Renames symmetric ground truth metrics by selecting the ground truth or alternative ground truth
    that is closest to the predicted positions.

    This function addresses the ambiguity in side-chain atom labeling due to chemical symmetries
    (e.g., the two carboxyl oxygens in Aspartate or the symmetric ring in Phenylalanine). It compares
    the predicted distances between symmetric (ambiguous) atoms and non-ambiguous atoms with the
    corresponding distances in both the standard ground truth and an alternative (swapped) ground
    truth. It then updates the ground truth positions and frames to the version that better
    matches the current model prediction, preventing the loss from penalizing correct structures
    that simply use a different labeling convention.

    Args:
        predicted_positions: Predicted 3D Cartesian coordinates for all atoms.
            Expected shape: (number_residues, 37, 3)
        ground_truth_transformation_matrix: Standard ground truth local transformation frames.
            Expected shape: (number_residues, 4, 4)
        ground_truth_positions: Standard ground truth 3D Cartesian coordinates for all atoms.
            Expected shape: (number_residues, 37, 3)
        alternative_ground_truth_transformation_matrix: Alternative ground truth local transformation frames,
            accounting for side-chain symmetries.
            Expected shape: (number_residues, 4, 4)
        alternative_ground_truth_positions: Alternative ground truth 3D Cartesian coordinates.
            Expected shape: (number_residues, 37, 3)
        sequence_amino_acid_labels: Integer indices representing the amino acid type for each residue.
            Expected shape: (number_residues)

    Returns:
        modified_ground_truth_positions: Selected ground truth positions (standard or alternative).
            Shape: (number_residues, 37, 3)
        modified_ground_truth_transformation_matrix: Selected ground truth transformation matrices.
            Shape: (number_residues, 4, 4)
    """
    # Important : We assume that there is no batch in our inputs

    # Get tensors that will be returned
    modified_ground_truth_positions = ground_truth_positions.clone()
    modified_ground_truth_transformation_matrix = ground_truth_transformation_matrix.clone()

    # Get non-ambiguous positions
    sequence_ambiguous_positions_masks = ambiguous_position_mask[sequence_amino_acid_labels]
    sequence_non_ambiguous_position_masks = ~sequence_ambiguous_positions_masks

    # Gets all the non ambigouous positions : (non_ambiguous_atoms_of_sequence, 3)
    sequence_unambiguous_predicted_positions = predicted_positions[sequence_non_ambiguous_position_masks]
    sequence_unambiguous_ground_truth_positions = ground_truth_positions[sequence_non_ambiguous_position_masks]

    # Go through all residues and only evaluate the amino acid residues with ambigous atoms
    for index, residue_index in enumerate(sequence_amino_acid_labels):

        # Skip if this residue has no ambiguous atoms.
        if not sequence_ambiguous_positions_masks[index].any():
            continue

        # Get current residue positions
        pred_res_pos = predicted_positions[index]
        gt_res_pos = ground_truth_positions[index]
        alt_gt_res_pos = alternative_ground_truth_positions[index]

        # Get current residue ambiguous positions
        pred_ambiguous_positions = pred_res_pos[ambiguous_position_mask[residue_index]]
        gt_ambiguous_positions = gt_res_pos[ambiguous_position_mask[residue_index]]
        alt_gt_ambiguous_positions = alt_gt_res_pos[ambiguous_position_mask[residue_index]]

        # Get the different distances
        # - predicitions<->predictions
        distance_predictions = torch.cdist(x1=pred_ambiguous_positions,
                                           x2=sequence_unambiguous_predicted_positions)

        # - ground_truth<->ground_truth
        distance_ground_truths = torch.cdist(x1=gt_ambiguous_positions,
                                             x2=sequence_unambiguous_ground_truth_positions)

        # - alternative_ground_truth <-> ground_truth
        distance_alternative_ground_truths = torch.cdist(x1=alt_gt_ambiguous_positions,
                                                         x2=sequence_unambiguous_ground_truth_positions)

        # Left element abs(predictions-alt_ground_truth)
        left_side = torch.sum(torch.abs(distance_predictions - distance_alternative_ground_truths))

        # Right element abs(predictions-ground_truth)
        right_side = torch.sum(torch.abs(distance_predictions - distance_ground_truths))

        # Modify the ground truth transformation matrix accordingly.
        if left_side < right_side:
            modified_ground_truth_positions[index] = alternative_ground_truth_positions[index]
            modified_ground_truth_transformation_matrix[index] = alternative_ground_truth_transformation_matrix[index]

    return modified_ground_truth_positions, modified_ground_truth_transformation_matrix


def compute_local_distance_difference_test(prediction_positions,
                                           ground_truth_positions,
                                           clamp_threshold=15.0,
                                           distance_thresholds=None):
    """
    Computes the Local Distance Difference Test (lDDT) score per residue.

    lDDT is a superposition-independent metric used to evaluate the local quality of a predicted
    protein structure. It compares the distances between all pairs of atoms in the prediction
    to the corresponding distances in the ground truth structure, provided those atoms are within
    a certain distance (the clamp threshold) in the ground truth. The score is calculated based
    on the fraction of preserved distances within several tolerance thresholds.

    In AlphaFold II, lDDT is primarily used as a validation metric and as the target for the
    confidence head (pLDDT). This implementation specifically focuses on Carbon Alpha (CA)
    atoms to assess the backbone quality.

    Args:
        prediction_positions: Predicted 3D Cartesian coordinates for all atoms.
            Expected shape: (*batch_dims, number_residues, 37, 3)
        ground_truth_positions: Ground truth 3D Cartesian coordinates for all atoms.
            Expected shape: (*batch_dims, number_residues, 37, 3)
        clamp_threshold: Maximum distance (in Angstroms) in the ground truth structure
            for an atom pair to be considered in the lDDT calculation.
        distance_thresholds: A list of tolerance thresholds (in Angstroms). A distance
            is considered "preserved" if the absolute difference between the predicted
            and ground truth distance is less than the threshold.

    Returns:
        local_difference_distance_test: The computed lDDT score for each residue.
            Shape: (*batch_dims, number_residues)
    """
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
    considered_ca_pair_counts = torch.sum(considered_ca_pairs.to(dtype), dim=-1)

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
        number_accurate_distances = torch.sum(current_indices.to(dtype), dim=-1)
        local_difference_distance_test += number_accurate_distances

    # Normalize the local difference distance test
    # Handle division by zero for residues with no neighbors within clamp_threshold
    normalization_factor = L * considered_ca_pair_counts
    local_difference_distance_test = torch.where(
        normalization_factor > 0,
        local_difference_distance_test / normalization_factor,
        torch.zeros_like(local_difference_distance_test)
    )

    return local_difference_distance_test
