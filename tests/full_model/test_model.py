import os
import math
import torch
from torch import nn
from pathlib import Path
from tests.utilities.testing_utilities import check_nn_module_method
from full_model.model import Model
from utilities.data.input import ModelInput


def test_model():
    number_cycles = 2
    number_evoformer_blocks = 3
    number_extra_msa_blocks = 4
    batch_size = 3
    msa_embedding = 6
    pair_representation_embedding = 7
    number_clusters = 10
    number_extra_sequences = 11
    single_representation_embedding = 384
    input_sequence_feature_dimension = 21
    msa_feature_dimension = 49
    input_extra_msa_feature_dimension = 25
    extra_msa_embedding = 19
    number_residues = 64  # Use a crop size

    device = torch.device('cpu')
    dtype = torch.float64
    reference_folder = os.path.join(os.path.dirname(__file__), 'reference_values')

    protein_id = "P90561"
    structure_path = os.path.join(reference_folder, f"structures/{protein_id}.npz")
    msa_path = os.path.join(reference_folder, f"raw_msa/{protein_id}.a3m")
    record_path = os.path.join(reference_folder, f"records/{protein_id}.json")

    model_input = ModelInput(structure_path=str(structure_path), msa_path=str(msa_path), record_path=str(record_path),
                             acceptance_slope_start=256, acceptance_slope_end=512,
                             residue_crop_size=None, emphasize_beginning_crops=False,
                             distribution_threshold=100,
                             maximum_cluster_sequences=number_clusters,
                             maximum_extra_msa_sequences=number_extra_sequences,
                             mask_probability=0.0,
                             device=device, dtype=dtype)

    data = model_input.get_data(number_samples=number_cycles, seed=42)

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
        device=device,
        dtype=dtype
    )

    simple_input_dict = {
        'input_msa_feature': data['input_msa_feature'][:, :number_residues, ...].unsqueeze(0),
        'input_extra_msa_feature': data['input_extra_msa_feature'][:, :number_residues, ...].unsqueeze(0),
        'input_sequence_feature': data['input_sequence_feature'][:number_residues, ...].unsqueeze(0),
        'input_residue_index_feature': data['input_residue_index_feature'][:number_residues, ...].unsqueeze(0),
        'ground_truth_frames': data['ground_truth_frames'][:number_residues, ...].unsqueeze(0),
        'alternative_ground_truth_frames': data['alternative_ground_truth_frames'][:number_residues, ...].unsqueeze(0),
        'ground_truth_angles': data['ground_truth_angles'][:number_residues, ...].unsqueeze(0),
        'alternative_ground_truth_angles': data['alternative_ground_truth_angles'][:number_residues, ...].unsqueeze(0),
        'ground_truth_global_positions': data['ground_truth_global_positions'][:number_residues, ...].unsqueeze(0),
        'alternative_ground_truth_global_positions': data['alternative_ground_truth_global_positions'][
            :number_residues, ...].unsqueeze(0),
        'distogram_labels': data['distogram_labels'][:number_residues, :number_residues, ...].unsqueeze(0),}

    batched_input_dict = {
        key: tensor.repeat(batch_size, *([1] * len(tensor.shape[1:]))) for key, tensor in simple_input_dict.items()
    }

    check_nn_module_method(
        module=model,
        input_tensor_dictionary=simple_input_dict,
        batched_input_tensor_dictionary=batched_input_dict,
        output_tensor_names=["model_angles",
                             "model_frames",
                             "model_final_positions",
                             "model_position_mask",
                             "model_pseudo_beta_positions",
                             "model_overall_fape_loss",
                             "model_auxillary_loss",
                             "model_predicted_lddt_loss",
                             "model_distogram_logits"],
        reference_folder=reference_folder,
        batch_size=batch_size,
        use_kwargs=False)
    print("Full Model Test Completed Successfully.")
