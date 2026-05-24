import math
import torch
from pathlib import Path

from embedders.recycling_embedder import RecyclingEmbedder
from tests.utilities.testing_utilities import check_nn_module_method


def test_embedder():
    batch_size = 3
    msa_embedding_dimension = 4
    pair_representation_dimension = 5
    number_clusters = 8
    number_residues = 35

    feature_shapes = {
        'previous_msa_representation': (number_clusters, number_residues, msa_embedding_dimension),
        'previous_pair_representation': (number_residues, number_residues, pair_representation_dimension),
        'previous_pseudo_carbon_beta_positions': (number_residues, 3),
    }

    output_tensor_names = [
        'msa_representation',
        'pair_representation_output',
    ]

    test_input_tensors = {
        key: torch.linspace(-2, 2, math.prod(shape)).reshape(shape).double()
        for key, shape in feature_shapes.items()
    }

    # Initialize the module with dummy dimensions
    recycling_embedder = RecyclingEmbedder(
        msa_embedding=msa_embedding_dimension,
        pair_representation_embedding=pair_representation_dimension,
        device=torch.device("cpu"),
        dtype=torch.float64
    )

    # Run the test check
    check_nn_module_method(
        module=recycling_embedder,
        input_tensor_dictionary=test_input_tensors,
        output_tensor_names=output_tensor_names,
        reference_folder=Path(__file__).parent / "recycling_embedder" / "reference_values",
        batch_size=batch_size
    )

    print(" - RecyclingEmbedder Test Completed Successfuly.")
