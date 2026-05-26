import os
import torch

from architecture_modules.structure_module.structure_module import StructureModule
from tests.utilities.testing_utilities import check_nn_module_method, get_structure_module_test_inputs


def test_structure_module_forward():
    device = torch.device('cpu')
    dtype = torch.float64

    config, combined_inputs = get_structure_module_test_inputs()

    reference_folder = os.path.join(os.path.dirname(__file__), 'reference_values')

    simple_inputs = {k: v[0] for k, v in combined_inputs.items()}
    batched_inputs = {k: v[1] for k, v in combined_inputs.items()}

    # Initialize the module
    module = StructureModule(
        single_representation_embedding=config["single_representation_embedding"],
        pair_representation_embedding=config["pair_representation_embedding"],
        device=device,
        dtype=dtype,
        number_layers=8,
        angle_representation_embedding=20,
        number_query_points=4,  # 4
        number_value_points=8,  # 8
        number_heads=12,  # 12
        head_embedding_dimension=16,  # 16
    )

    input_tensor_dictionary = {
        "single_representation": simple_inputs["single_representation"].to(device),
        "pair_representation": simple_inputs["pair_representation"].to(device),
        "sequence_amino_acid_labels": simple_inputs["sequence_amino_acid_labels"].to(device),
    }

    batched_input_tensor_dictionary = {
        "single_representation": batched_inputs["single_representation"].to(device),
        "pair_representation": batched_inputs["pair_representation"].to(device),
        "sequence_amino_acid_labels": batched_inputs["sequence_amino_acid_labels"].to(device),
    }

    output_tensor_names = [
        "structure_module_angles",
        "structure_module_frames",
        "structure_module_final_positions",
        "structure_module_position_mask",
        "structure_module_pseudo_beta_positions"
    ]

    check_nn_module_method(
        module=module,
        input_tensor_dictionary=input_tensor_dictionary,
        output_tensor_names=output_tensor_names,
        reference_folder=reference_folder,
        batch_size=config["batch_size"],
        batched_input_tensor_dictionary=batched_input_tensor_dictionary
    )
    print(" - StructureModule Test Completed Successfuly.")
