import os
import torch

from architecture_modules.structure_module.invariant_point_attention_module import InvariantPointAttention
from tests.utilities.testing_utilities import check_nn_module_method, get_structure_module_test_inputs


def test_invariant_point_attention_forward():
    device = torch.device('cpu')
    dtype = torch.float64

    config, combined_inputs = get_structure_module_test_inputs()

    reference_folder = os.path.join(os.path.dirname(__file__), 'reference_values')

    simple_inputs = {k: v[0] for k, v in combined_inputs.items()}
    batched_inputs = {k: v[1] for k, v in combined_inputs.items()}

    module = InvariantPointAttention(
        single_representation_embedding=config["single_representation_embedding"],
        pair_representation_embedding=config["pair_representation_embedding"],
        number_query_points=config["number_query_points"],
        number_value_points=config["number_value_points"],
        number_heads=config["number_heads"],
        head_embedding_dimension=config["head_embedding_dimension"],
        device=device,
        dtype=dtype
    )

    input_tensor_dictionary = {
        "single_representation": simple_inputs["single_representation"].to(device),
        "pair_representation": simple_inputs["pair_representation"].to(device),
        "transformation_matrix": simple_inputs["transformation_matrix"].to(device),
    }

    batched_input_tensor_dictionary = {
        "single_representation": batched_inputs["single_representation"].to(device),
        "pair_representation": batched_inputs["pair_representation"].to(device),
        "transformation_matrix": batched_inputs["transformation_matrix"].to(device),
    }

    output_tensor_names = ["ipa_out"]

    check_nn_module_method(
        module=module,
        input_tensor_dictionary=input_tensor_dictionary,
        output_tensor_names=output_tensor_names,
        reference_folder=reference_folder,
        batch_size=config["batch_size"],
        batched_input_tensor_dictionary=batched_input_tensor_dictionary
    )
    print(" - InvariantPointAttention Test Completed Successfuly.")