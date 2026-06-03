import math
import torch
from pathlib import Path

from architecture_modules.lddt_module.lddt_module import LddtModule
from tests.utilities.testing_utilities import check_nn_module_method


def test_lddt_module_parity():
    """Verify that the LddtModule outputs match pre-computed reference values."""
    batch_size = 3
    number_residues = 10
    single_representation_embedding = 11
    intermediate_embedding = 6

    shape = (number_residues, single_representation_embedding)

    # Deterministic input for single_representation
    test_inputs = {'single_representation': torch.linspace(-3 / 5, 3 / 5, math.prod(shape)).reshape(shape).double()}

    module = LddtModule(single_representation_embedding=single_representation_embedding,
                        intermediate_embedding=intermediate_embedding,
                        device=torch.device('cpu'),
                        dtype=torch.float64)

    check_nn_module_method(module=module,
                           input_tensor_dictionary=test_inputs,
                           output_tensor_names=["lddt_logits", "lddt_probabilities", "predicted_lddt_per_residue"],
                           reference_folder=Path(__file__).parent / "reference_values",
                           batch_size=batch_size)


def test_lddt_module_initialization():
    """Verify that lddt_bins is properly registered as a buffer."""
    module = LddtModule(single_representation_embedding=11,
                        intermediate_embedding=6,
                        device=torch.device('cpu'),
                        dtype=torch.float64)

    assert hasattr(module, "lddt_bins")
    # Verify shape and contents
    assert module.lddt_bins.shape == (50,)
    expected_bins = torch.arange(start=1, end=100, step=2, dtype=torch.float64)
    torch.testing.assert_close(module.lddt_bins, expected_bins)

    # Verify it is in the state_dict (which means it's a registered buffer/parameter)
    assert "lddt_bins" in module.state_dict()
    # Verify it doesn't require gradients
    assert not module.lddt_bins.requires_grad
