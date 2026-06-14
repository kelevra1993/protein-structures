import torch
import math
import os
from torch import nn
from pathlib import Path
from typing import List, Dict, Tuple

from utilities.tensor_utilities import print_tensor_shape


def check_nn_module_method(module: nn.Module, input_tensor_dictionary: Dict[str, torch.Tensor],
                           output_tensor_names: List[str],
                           reference_folder: str | Path, batch_size: int,
                           batched_input_tensor_dictionary: Dict[str, torch.Tensor] = None,
                           use_kwargs=True):
    """
    Performs a deterministic regression test on a PyTorch module's forward pass.

    This utility ensures architectural consistency by comparing a module's output 
    against saved reference tensors. It eliminates non-determinism by:
    1. Setting the module to evaluation mode (disabling Dropout, BatchNorm, etc.).
    2. Converting the module and inputs to double precision (float64) to minimize numerical drift.
    3. Overwriting all module parameters with a fixed, deterministic sequence (torch.linspace).

    The function validates both a single-instance "simple" pass and a "batched" pass 
    using broadcasting to ensure the module correctly handles batch dimensions.

    :param module: The nn.Module to be tested.
    :param input_tensor_dictionary: A dictionary mapping input argument names to their 
                                    respective input tensors.
    :param output_tensor_names: A list of names for the expected output tensors, 
                                matching the filenames in the reference_folder.
    :param reference_folder: Path to the directory containing the '.pt' reference files.
    :param batch_size: The size of the batch to simulate for the batched consistency check.
    :param batched_input_tensor_dictionary: Optional dictionary of pre-batched inputs. If None, it broadcasts the simple inputs.
    """
    # First We Just Modify The Parameters Of The Module To Be Fixed
    # Set To eval() To Disable Dropout Or Any Batch Normalisation (anything non deterministic)
    module.eval()
    module.double()

    with torch.no_grad():
        for param in module.parameters():
            # Set to the same parameters.
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

    # Module Is Now Ready For Testing

    # Simple Input And Batched Input
    simple_input_dictionary = {input_name: input_tensor for input_name, input_tensor in input_tensor_dictionary.items()}
    if batched_input_tensor_dictionary is not None:
        batched_input_dictionary = batched_input_tensor_dictionary
    else:
        batched_input_dictionary = {input_name: input_tensor.broadcast_to((batch_size,) + input_tensor.shape)
                                    for input_name, input_tensor in input_tensor_dictionary.items()}

    # Simple Output And Batched Output
    if use_kwargs:
        simple_output = module(**simple_input_dictionary)
        batched_output = module(**batched_input_dictionary)
    else:
        simple_output = module(simple_input_dictionary).values()
        batched_output = module(batched_input_dictionary).values()

    if isinstance(simple_output, torch.Tensor):
        simple_output = [simple_output]

    if isinstance(batched_output, torch.Tensor):
        batched_output = [batched_output]

    out_file_names = [f'{reference_folder}/{out_name}.pt' for out_name in output_tensor_names]
    batched_out_file_names = [f'{reference_folder}/{out_name}_batched.pt' for out_name in output_tensor_names]

    for output_tensor, batched_output_tensor, output_filename, batched_output_filename, output_tensor_name in zip(
            simple_output, batched_output,
            out_file_names, batched_out_file_names,
            output_tensor_names):

        expected_output_tensor = torch.load(output_filename, weights_only=True)
        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'

        if os.path.exists(batched_output_filename):
            expected_batched_output_tensor = torch.load(batched_output_filename, weights_only=True)
        else:
            expected_out_batch_shape = (batch_size,) + expected_output_tensor.shape
            expected_batched_output_tensor = expected_output_tensor.unsqueeze(0).broadcast_to(expected_out_batch_shape)

        assert torch.allclose(batched_output_tensor, expected_batched_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Batched Check.'


def get_evoformer_test_inputs() -> Tuple[Dict[str, int], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Generates deterministic test inputs and configurations for the Evoformer stack modules.

    This utility creates standardized, reproducible tensors (`msa_representation` and 
    `pair_representation`) initialized via `torch.linspace`. It returns both the simple 
    tensors and their broadcasted batched equivalents, alongside the structural dimension 
    configurations required to instantiate the modules (e.g., embedding dimensions, heads).

    Returns:
        Tuple containing:
            - A configuration dictionary mapping dimension names (str) to their integer sizes.
            - A test inputs dictionary mapping tensor names (str) to a tuple containing 
              (simple_tensor, batched_tensor). Both tensors are cast to float64.
    """
    batch_size = 3
    msa_embedding = 4
    pair_representation_embedding = 5
    head_embedding_dimension = 6
    channel_scaler = 3
    number_heads = 7
    number_sequences = 8
    number_residues = 9
    intermediate_embedding = 6
    triangle_multiplication_embedding = 128

    test_msa_representation_shape = (number_sequences, number_residues, msa_embedding)
    test_pair_representation_shape = (number_residues, number_residues, pair_representation_embedding)
    test_msa_representation_shape_batched = (batch_size,) + test_msa_representation_shape
    test_pair_representation_shape_batched = (batch_size,) + test_pair_representation_shape

    test_msa_representation = torch.linspace(-2, 2, math.prod(test_msa_representation_shape)).reshape(
        test_msa_representation_shape)
    test_pair_representation = torch.linspace(-2, 2, math.prod(test_pair_representation_shape)).reshape(
        test_pair_representation_shape)
    test_msa_representation_batch = torch.linspace(-2, 2, math.prod(test_msa_representation_shape_batched)).reshape(
        test_msa_representation_shape_batched)
    test_pair_representation_batch = torch.linspace(-2, 2, math.prod(test_pair_representation_shape_batched)).reshape(
        test_pair_representation_shape_batched)

    config = {
        "batch_size": batch_size,
        "msa_embedding": msa_embedding,
        "pair_representation_embedding": pair_representation_embedding,
        "head_embedding_dimension": head_embedding_dimension,
        "channel_scaler": channel_scaler,
        "intermediate_embedding": intermediate_embedding,
        "number_heads": number_heads,
        "number_sequences": number_sequences,
        "number_residues": number_residues,
        "triangle_multiplication_embedding": triangle_multiplication_embedding,
    }

    test_inputs = {
        "msa_representation": (test_msa_representation.double(), test_msa_representation_batch.double()),
        "pair_representation": (test_pair_representation.double(), test_pair_representation_batch.double()),
    }

    return config, test_inputs


def get_structure_module_test_inputs() -> Tuple[Dict[str, int], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Generates deterministic test inputs and configurations for the Structure Module.

    This utility creates standardized, reproducible tensors initialized via `torch.linspace`.
    It returns both the simple tensors and their broadcasted batched equivalents, alongside
    the structural dimension configurations required to instantiate the modules.

    Returns:
        Tuple containing:
            - A configuration dictionary mapping dimension names (str) to their integer sizes.
            - A test inputs dictionary mapping tensor names (str) to a tuple containing
              (simple_tensor, batched_tensor). Both tensors are cast to float64.
    """
    number_layers = 2
    batch_size = 3
    msa_embedding_dimension = 4
    pair_representation_embedding = 5
    head_embedding_dimension = 6
    number_heads = 7
    number_sequences = 8
    number_extra_sequences = 9
    number_residues = 10
    single_representation_embedding = 11
    number_query_points = 12
    number_value_points = 13
    number_torsion_angles = 7

    feature_shapes = {
        'msa_representation': (number_sequences, number_residues, msa_embedding_dimension),
        'single_representation': (number_residues, single_representation_embedding),
        'pair_representation': (number_residues, number_residues, pair_representation_embedding),
        'residue_index': (number_residues,),
        'positions': (number_residues, 3),
        'query_tensor': (number_heads, number_residues, head_embedding_dimension),
        'key_tensor': (number_heads, number_residues, head_embedding_dimension),
        'value_tensor': (number_heads, number_residues, head_embedding_dimension),
        'query_point_tensor': (number_heads, number_query_points, number_residues, 3),
        'key_point_tensor': (number_heads, number_query_points, number_residues, 3),
        'value_point_tensor': (number_heads, number_value_points, number_residues, 3),
        'transformation_matrix': (number_residues, 4, 4),
        'att_scores': (number_heads, number_residues, number_residues),
        'angle_representation': (number_residues, head_embedding_dimension),
        's_initial': (number_residues, single_representation_embedding),
        'residue_angles': (number_residues, number_torsion_angles, 2),
        'sequence_amino_acid_labels': (number_residues,),
    }

    config = {
        "number_layers": number_layers,
        "batch_size": batch_size,
        "msa_embedding_dimension": msa_embedding_dimension,
        "pair_representation_embedding": pair_representation_embedding,
        "head_embedding_dimension": head_embedding_dimension,
        "number_heads": number_heads,
        "number_sequences": number_sequences,
        "number_extra_sequences": number_extra_sequences,
        "number_residues": number_residues,
        "single_representation_embedding": single_representation_embedding,
        "number_query_points": number_query_points,
        "number_value_points": number_value_points,
        "number_torsion_angles": number_torsion_angles,
    }

    test_inputs = {
        key: torch.linspace(-2 - i / 5, 2 + i / 5, math.prod(shape)).reshape(shape).double()
        for i, (key, shape) in enumerate(feature_shapes.items())
    }

    test_inputs['sequence_amino_acid_labels'] = torch.arange(test_inputs['sequence_amino_acid_labels'].numel()).reshape(
        test_inputs['sequence_amino_acid_labels'].shape) % 20

    batched_test_inputs = {
        key: tensor.unsqueeze(0).broadcast_to((batch_size,) + tensor.shape)
        for key, tensor in test_inputs.items()
    }

    combined_inputs = {
        key: (test_inputs[key], batched_test_inputs[key])
        for key in feature_shapes.keys()
    }

    return config, combined_inputs
