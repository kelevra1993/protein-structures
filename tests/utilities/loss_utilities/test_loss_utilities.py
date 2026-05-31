import os
import pytest
import torch
import math
from utilities.loss_utilities import compute_fape_loss

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
