import os
import torch
import math

from utilities.constants import atom_to_index
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss, compute_distogram_loss, \
    compute_local_distance_difference_test


def test_compute_torsion_angle_loss_identical_inputs():
    """Verify that identical inputs with unit norm yield zero loss."""
    batch_size = 2
    number_residues = 10

    # Angles on the unit circle (cos=1, sin=0)
    ground_truth_angles = torch.zeros(batch_size, number_residues, 7, 2)
    ground_truth_angles[..., 0] = 1.0

    # Predictions match ground truth and are already normalized
    predicted_unnormalised_angles = ground_truth_angles.clone()

    loss = compute_torsion_angle_loss(predicted_unnormalised_angles=predicted_unnormalised_angles,
                                      ground_truth_angles=ground_truth_angles,
                                      alternative_ground_truth_angles=ground_truth_angles.clone())

    torch.testing.assert_close(loss, torch.zeros(batch_size))


def test_compute_torsion_angle_loss_alternative_ground_truth():
    """Verify that the loss handles side-chain symmetry by picking the minimum distance."""
    batch_size = 1
    number_residues = 5

    # Primary Ground Truth: (1, 0)
    ground_truth_angles = torch.zeros(batch_size, number_residues, 7, 2)
    ground_truth_angles[..., 0] = 1.0

    # Alternative Ground Truth: (0, 1)
    alternative_ground_truth_angles = torch.zeros(batch_size, number_residues, 7, 2)
    alternative_ground_truth_angles[..., 1] = 1.0

    # Prediction matches Alternative Ground Truth perfectly
    predicted_unnormalised_angles = alternative_ground_truth_angles.clone()

    loss = compute_torsion_angle_loss(predicted_unnormalised_angles=predicted_unnormalised_angles,
                                      ground_truth_angles=ground_truth_angles,
                                      alternative_ground_truth_angles=alternative_ground_truth_angles)

    # Torsion loss should be 0 because it picked the alternative_ground_truth_angles
    # Norm loss should be 0 because predicted_unnormalised_angles is normalized
    torch.testing.assert_close(loss, torch.zeros(batch_size))


def test_compute_torsion_angle_loss_normalization_penalty():
    """Verify that unnormalized predictions trigger the angle normalization penalty."""
    batch_size = 1
    number_residues = 1
    angle_norm_loss_scaler = 0.5

    # Ground Truth is (1, 0)
    ground_truth_angles = torch.zeros(batch_size, number_residues, 7, 2)
    ground_truth_angles[..., 0] = 1.0

    # Prediction points in same direction (1, 0) but has norm of 2.0
    predicted_unnormalised_angles = torch.zeros(batch_size, number_residues, 7, 2)
    predicted_unnormalised_angles[..., 0] = 2.0

    loss = compute_torsion_angle_loss(predicted_unnormalised_angles=predicted_unnormalised_angles,
                                      ground_truth_angles=ground_truth_angles,
                                      alternative_ground_truth_angles=ground_truth_angles.clone(),
                                      angle_norm_loss_scaler=angle_norm_loss_scaler)

    # Torsion loss: prediction will be normalized to (1, 0), so distance to Ground Truth is 0.
    # Norm loss: mean(|2 - 1|) = 1.0
    # Total loss: 0 + 0.5 * 1.0 = 0.5
    expected_loss = torch.tensor([0.5])
    torch.testing.assert_close(loss, expected_loss)


def generate_test_inputs(batch_shape, number_residues):
    """Helper to generate normalized ground truth, alternative, and predicted angles."""
    ground_truth = torch.randn(*batch_shape, number_residues, 7, 2)
    ground_truth = torch.nn.functional.normalize(ground_truth, dim=-1)
    return ground_truth, ground_truth.clone(), ground_truth.clone()


def test_compute_torsion_angle_loss_batch_shapes():
    """Verify that torsion loss handles various batch dimensions correctly."""
    number_residues = 4

    # No batch shape
    ground_truth, alternative, predicted = generate_test_inputs((), number_residues)
    loss_unbatched = compute_torsion_angle_loss(predicted, ground_truth, alternative)
    assert loss_unbatched.shape == ()

    # 1D batch shape
    batch_size_1d = 3
    ground_truth, alternative, predicted = generate_test_inputs((batch_size_1d,), number_residues)
    loss_1d = compute_torsion_angle_loss(predicted, ground_truth, alternative)
    assert loss_1d.shape == (batch_size_1d,)

    # 2D batch shape
    batch_size_2d_1, batch_size_2d_2 = 2, 5
    ground_truth, alternative, predicted = generate_test_inputs((batch_size_2d_1, batch_size_2d_2), number_residues)
    loss_2d = compute_torsion_angle_loss(predicted, ground_truth, alternative)
    assert loss_2d.shape == (batch_size_2d_1, batch_size_2d_2)


def test_compute_fape_loss_known_values():
    """Verify loss matches a pre-computed reference value for parity."""
    batch_size = 2
    number_residues = 15

    pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
    gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)

    for k in range(number_residues):
        pred_tr_m[k, :3, -1] += k + 1
        gt_tr_m[k, :3, -1] += k + 2

    pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
    gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))

    pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
        (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size,
                                                                                                           1, 1)

    gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
        (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size,
                                                                                                            1, 1)

    fape_loss = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                                  predicted_positions=pred_carbon_alpha_positions,
                                  ground_truth_transformation_matrix=gt_tr_m,
                                  ground_truth_positions=gt_carbon_alpha_positions,
                                  length_scaler=2,
                                  epsilon=2e-4,
                                  distance_clamp=2.0)

    current_dir = os.path.dirname(__file__)
    reference_path = os.path.join(current_dir, "reference_values", "fape_loss_scratchpad.pt")
    expected_loss = torch.load(reference_path, weights_only=True)

    torch.testing.assert_close(fape_loss, expected_loss)


def test_compute_fape_loss_identical_inputs():
    """Verify that identical inputs yield the baseline epsilon-based loss."""
    batch_size = 2
    number_residues = 15
    length_scaler = 10
    epsilon = 1e-4

    pred_tr_m = torch.eye(4).unsqueeze(dim=0).repeat(batch_size, number_residues, 1, 1)
    pred_pos = torch.randn(batch_size, number_residues, 3)

    fape_loss = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                                  predicted_positions=pred_pos,
                                  ground_truth_transformation_matrix=pred_tr_m.clone(),
                                  ground_truth_positions=pred_pos.clone(),
                                  length_scaler=length_scaler,
                                  epsilon=epsilon)

    expected_loss_value = math.sqrt(epsilon) / length_scaler
    expected_loss = torch.full((batch_size,), expected_loss_value)

    torch.testing.assert_close(fape_loss, expected_loss)


def test_compute_fape_loss_batch_shapes():
    """Verify the loss computation handles arbitrary batch dimensions correctly."""
    number_residues = 15
    length_scaler = 10

    # No batch shape
    tr_m_no_batch = torch.eye(4).unsqueeze(0).repeat(number_residues, 1, 1)
    pos_no_batch = torch.randn(number_residues, 3)

    loss_no_batch = compute_fape_loss(predicted_transformation_matrix=tr_m_no_batch,
                                      predicted_positions=pos_no_batch,
                                      ground_truth_transformation_matrix=tr_m_no_batch.clone(),
                                      ground_truth_positions=pos_no_batch.clone(),
                                      length_scaler=length_scaler)
    assert loss_no_batch.shape == ()

    # 1D batch shape
    batch_size_1d = 3
    tr_m_1d = tr_m_no_batch.unsqueeze(0).repeat(batch_size_1d, 1, 1, 1)
    pos_1d = pos_no_batch.unsqueeze(0).repeat(batch_size_1d, 1, 1)

    loss_1d = compute_fape_loss(predicted_transformation_matrix=tr_m_1d,
                                predicted_positions=pos_1d,
                                ground_truth_transformation_matrix=tr_m_1d.clone(),
                                ground_truth_positions=pos_1d.clone(),
                                length_scaler=length_scaler)
    assert loss_1d.shape == (batch_size_1d,)

    # 2D batch shape
    batch_size_2d_1, batch_size_2d_2 = 2, 4
    tr_m_2d = tr_m_no_batch.unsqueeze(0).unsqueeze(0).repeat(batch_size_2d_1, batch_size_2d_2, 1, 1, 1)
    pos_2d = pos_no_batch.unsqueeze(0).unsqueeze(0).repeat(batch_size_2d_1, batch_size_2d_2, 1, 1)

    loss_2d = compute_fape_loss(predicted_transformation_matrix=tr_m_2d,
                                predicted_positions=pos_2d,
                                ground_truth_transformation_matrix=tr_m_2d.clone(),
                                ground_truth_positions=pos_2d.clone(),
                                length_scaler=length_scaler)
    assert loss_2d.shape == (batch_size_2d_1, batch_size_2d_2)


def test_compute_fape_loss_clamping():
    """Verify that the loss correctly applies the distance clamp limit."""
    batch_size = 2
    number_residues = 15
    distance_clamp = 2.0
    length_scaler = 1

    # Identical transformations
    tr_m = torch.eye(4).unsqueeze(0).repeat(batch_size, number_residues, 1, 1)

    # Very distant positions to ensure clamp is triggered
    pred_pos = torch.zeros(batch_size, number_residues, 3)
    gt_pos = torch.ones(batch_size, number_residues, 3) * 100.0  # Far away

    fape_loss = compute_fape_loss(predicted_transformation_matrix=tr_m,
                                  predicted_positions=pred_pos,
                                  ground_truth_transformation_matrix=tr_m,
                                  ground_truth_positions=gt_pos,
                                  length_scaler=length_scaler,
                                  distance_clamp=distance_clamp)

    # If clamped at distance_clamp, the mean should be distance_clamp / length_scaler
    expected_loss = torch.full((batch_size,), distance_clamp / length_scaler)

    torch.testing.assert_close(fape_loss, expected_loss)


def test_compute_distogram_loss():
    """Verify that the distogram loss correctly computes cross-entropy."""
    batch_size = 2
    number_residues = 10
    number_bins = 64

    # Create dummy logits: (batch_size, number_residues, number_residues, number_bins)
    # We want a very small loss, so we'll make the ground truth category have high logits
    logits = torch.randn(batch_size, number_residues, number_residues, number_bins)

    # Labels: (batch_size, number_residues, number_residues)
    labels = torch.randint(low=0, high=number_bins, size=(batch_size, number_residues, number_residues))

    # Set logits for the correct labels to be very high to ensure the loss is near zero
    for b in range(batch_size):
        for i in range(number_residues):
            for j in range(number_residues):
                logits[b, i, j, labels[b, i, j]] = 100.0

    loss = compute_distogram_loss(distogram_logits=logits, distogram_labels=labels)

    # Loss should be effectively zero
    assert loss.item() < 1e-4
    assert loss.shape == ()


def test_lddt_shape():
    """Verify that the lDDT output shape matches (batch_size, number_residues)."""
    batch_size = 2
    number_residues = 10
    prediction_positions = torch.randn((batch_size, number_residues, 37, 3))
    ground_truth_positions = torch.randn((batch_size, number_residues, 37, 3))

    lddt = compute_local_distance_difference_test(prediction_positions=prediction_positions,
                                                  ground_truth_positions=ground_truth_positions)

    assert lddt.shape == (batch_size, number_residues)


def test_lddt_perfect_prediction():
    """Verify that identical structures yield a perfect lDDT score of 1.0."""
    batch_size = 1
    number_residues = 5
    # Create CA positions in a line: (0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)
    # All pairs are within 15A clamp threshold
    positions = torch.zeros((batch_size, number_residues, 37, 3))
    ca_idx = atom_to_index["CA"]
    for i in range(number_residues):
        positions[:, i, ca_idx, 0] = float(i)

    lddt = compute_local_distance_difference_test(prediction_positions=positions,
                                                  ground_truth_positions=positions)

    # All residues should have score 1.0 since they all have at least one neighbor
    torch.testing.assert_close(lddt, torch.ones_like(lddt))


def test_lddt_complete_mismatch():
    """Verify that structures with large distance errors yield an lDDT score of 0.0."""
    batch_size = 1
    number_residues = 5
    ca_idx = atom_to_index["CA"]

    # Ground truth: line at x=0, 1, 2, 3, 4
    # Prediction: line at x=0, 10, 20, 30, 40 (distance differences > 4.0A)
    gt_positions = torch.zeros((batch_size, number_residues, 37, 3))
    pred_positions = torch.zeros((batch_size, number_residues, 37, 3))
    for i in range(number_residues):
        gt_positions[:, i, ca_idx, 0] = float(i)
        pred_positions[:, i, ca_idx, 0] = float(i * 10)

    lddt = compute_local_distance_difference_test(prediction_positions=pred_positions,
                                                  ground_truth_positions=gt_positions)

    # All residues should have score 0.0
    torch.testing.assert_close(lddt, torch.zeros_like(lddt))


def test_lddt_clamp_threshold():
    """Verify that atom pairs beyond the clamp threshold are ignored in lDDT."""
    batch_size = 1
    number_residues = 2
    ca_idx = atom_to_index["CA"]

    # Residue 0 and 1 are 20A apart (clamped at 15A)
    gt_positions = torch.zeros((batch_size, number_residues, 37, 3))
    gt_positions[:, 1, ca_idx, 0] = 20.0

    # Prediction matches exactly
    pred_positions = gt_positions.clone()

    lddt = compute_local_distance_difference_test(prediction_positions=pred_positions,
                                                  ground_truth_positions=gt_positions,
                                                  clamp_threshold=15.0)

    # Since they are 20A apart, they have NO neighbors within 15A.
    # The score should be 0.0 (due to our division by zero fix)
    torch.testing.assert_close(lddt, torch.zeros_like(lddt))


def test_lddt_distance_thresholds_accuracy():
    """Verify that lDDT correctly scores partially preserved distances based on thresholds."""
    batch_size = 1
    number_residues = 2
    ca_idx = atom_to_index["CA"]

    # Ground Truth : Two residues 5A apart
    # Prediction is 6.5A apart (error of 1.5A)
    # Thresholds: [0.5, 1.0, 2.0, 4.0]
    # 1.5 < 0.5: No
    # 1.5 < 1.0: No
    # 1.5 < 2.0: Yes
    # 1.5 < 4.0: Yes
    # Expected score: 2/4 = 0.5
    gt_positions = torch.zeros((batch_size, number_residues, 37, 3))
    pred_positions = torch.zeros((batch_size, number_residues, 37, 3))
    gt_positions[:, 1, ca_idx, 0] = 5.0
    pred_positions[:, 1, ca_idx, 0] = 6.5

    lddt = compute_local_distance_difference_test(prediction_positions=pred_positions,
                                                  ground_truth_positions=gt_positions)

    expected_lddt = torch.tensor([[0.5, 0.5]])
    torch.testing.assert_close(lddt, expected_lddt)


def test_lddt_isolated_residue():
    """Verify that residues with no neighbors within the clamp threshold yield a score of 0.0."""
    batch_size = 1
    number_residues = 1
    # Single residue structure - no neighbors possible
    positions = torch.zeros((batch_size, number_residues, 37, 3))

    lddt = compute_local_distance_difference_test(prediction_positions=positions, ground_truth_positions=positions)

    # Should be 0.0 instead of NaN
    assert not torch.isnan(lddt).any()
    torch.testing.assert_close(lddt, torch.zeros_like(lddt))
