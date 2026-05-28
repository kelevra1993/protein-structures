import math
import torch
from torch import nn
from pathlib import Path
from tests.utilities.testing_utilities import check_nn_module_method
from full_model.model import Model

number_cycles = 2
number_evoformer_blocks = 3
number_extra_msa_blocks = 4
batch_size = 5
msa_embedding = 6
pair_representation_embedding = 7
head_embedding_dimension = 8
number_heads = 9
number_clusters = 10
number_extra_sequences = 11
number_residues = 12
single_representation_embedding = 384
number_query_points = 14
number_value_points = 15
number_torsion_angles = 7
input_sequence_feature_dimension = 21
msa_feature_dimension = 49
input_extra_msa_feature_dimension = 18
extra_msa_embedding = 19

feature_shapes = {
    'input_msa_feature': (number_clusters, number_residues, msa_feature_dimension, number_cycles),
    'input_sequence_feature': (number_residues, input_sequence_feature_dimension, number_cycles),
    'input_residue_index_feature': (number_residues, number_cycles),
    'input_extra_msa_feature': (number_extra_sequences, number_residues, input_extra_msa_feature_dimension,
                                number_cycles),
}

test_inputs = {
    key: torch.linspace(-2 - i / 5, 2 + i / 5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['input_residue_index_feature'] = torch.arange(number_residues).view(number_residues, 1).broadcast_to(
    number_residues, number_cycles)
test_inputs['input_sequence_feature'] = nn.functional.one_hot(torch.arange(number_residues) % 20,
                                                              num_classes=input_sequence_feature_dimension).double()
test_inputs['input_sequence_feature'] = test_inputs['input_sequence_feature'].unsqueeze(-1).broadcast_to(
    feature_shapes['input_sequence_feature'])


def test_model():
    configuration = {
        'GlobalConfiguration': {
            'msa_embedding': msa_embedding,
            'pair_representation_embedding': pair_representation_embedding,
            'extra_msa_embedding': extra_msa_embedding,
            'single_representation_embedding': single_representation_embedding,
            'input_sequence_feature_dimension': input_sequence_feature_dimension,
            'input_msa_feature_dimension': msa_feature_dimension,
            'input_extra_msa_feature_dimension': input_extra_msa_feature_dimension
        },
        'InputEmbedder': {
            'number_neighbouring_amino_acids': 32
        },
        'ExtraMsaStack': {
            'number_blocks': number_extra_msa_blocks,
            'msa_number_heads': 8,
            'msa_head_embedding_dimension': 8,
            'msa_global_number_heads': 8,
            'msa_global_head_embedding_dimension': 8,
            'pair_number_heads': 4,
            'pair_head_embedding_dimension': 32,
            'intermediate_embedding': 32,
            'msa_transition_channel_scaler': 4,
            'pair_stack_channel_scaler': 4,
            'triangle_multiplication_embedding': 128
        },
        'EvoformerStack': {
            'number_blocks': number_evoformer_blocks,
            'msa_number_heads': 8,
            'msa_head_embedding_dimension': 32,
            'pair_number_heads': 4,
            'pair_head_embedding_dimension': 32,
            'msa_transition_channel_scaler': 4,
            'pair_stack_channel_scaler': 4,
            'intermediate_embedding': 32,
            'triangle_multiplication_embedding': 128
        },
        'StructureModule': {
            'number_structure_module_iterations': 8,
            'angle_representation_embedding': 128,
            'number_heads': 12,
            'head_embedding_dimension': 16,
            'number_query_points': 4,
            'number_value_points': 8,
            'number_torsion_angles': 7
        }
    }
    model = Model(
        configuration=configuration,
        device=torch.device('cpu'),
        dtype=torch.float64
    )

    check_nn_module_method(
        module=model,
        input_tensor_dictionary=test_inputs,
        output_tensor_names=["model_angles",
                             "model_frames",
                             "model_final_positions",
                             "model_position_mask",
                             "model_pseudo_beta_positions"],
        reference_folder=Path(__file__).parent / "reference_values",
        batch_size=batch_size,
        use_kwargs=False)
    print("Full Model Test Completed Successfully.")
