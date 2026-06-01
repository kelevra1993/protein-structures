import torch
from architecture_modules.distogram_module import DistogramModule
from utilities.tensor_utilities import get_device


def test_distogram_module_output_shape():
    """Verify that the distogram module produces the correct output tensor shape."""
    device = get_device()
    dtype = torch.float32
    pair_representation_embedding = 128
    number_residues = 16
    batch_size = 2

    module = DistogramModule(
        pair_representation_embedding=pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    pair_representation = torch.randn(
        batch_size, number_residues, number_residues, pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    logits, output = module(pair_representation)

    # Check shape: (batch_size, number_residues, number_residues, 64)
    assert logits.shape == (batch_size, number_residues, number_residues, 64)
    assert output.shape == (batch_size, number_residues, number_residues, 64)


def test_distogram_module_symmetrization():
    """Verify that the predicted distogram is symmetric across residue pairs."""
    device = get_device()
    dtype = torch.float32
    pair_representation_embedding = 64
    number_residues = 10

    module = DistogramModule(
        pair_representation_embedding=pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    # Create an asymmetric pair representation
    pair_representation = torch.randn(
        number_residues, number_residues, pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    logits, output = module(pair_representation)

    # The output distogram (probabilities) should be symmetric: dist(i, j) == dist(j, i)
    # Output shape: (number_residues, number_residues, 64)
    output_transposed = output.transpose(0, 1)
    
    torch.testing.assert_close(output, output_transposed, atol=1e-6, rtol=1e-6)


def test_distogram_module_probability_normalization():
    """Verify that the predicted distogram bins sum to one for each residue pair."""
    device = get_device()
    dtype = torch.float32
    pair_representation_embedding = 32
    number_residues = 8

    module = DistogramModule(
        pair_representation_embedding=pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    pair_representation = torch.randn(
        number_residues, number_residues, pair_representation_embedding,
        device=device,
        dtype=dtype
    )

    logits, output = module(pair_representation)

    # Probabilities should sum to 1.0 along the last dimension
    sum_probabilities = torch.sum(output, dim=-1)
    expected_sum = torch.ones_like(sum_probabilities)
    
    torch.testing.assert_close(sum_probabilities, expected_sum, atol=1e-6, rtol=1e-6)
