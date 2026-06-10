import torch
from pathlib import Path
from full_model.model import Model
from utilities.data.input import ModelInput

device = torch.device('cpu')
dtype = torch.float64
reference_folder = Path("tests/full_model/reference_values")

configuration = {
    'GlobalConfiguration': {
        'msa_embedding': 6, 'pair_representation_embedding': 7,
        'extra_msa_embedding': 19, 'single_representation_embedding': 384,
        'input_sequence_feature_dimension': 21, 'input_msa_feature_dimension': 49,
        'input_extra_msa_feature_dimension': 25
    },
    'InputEmbedder': {'number_neighbouring_amino_acids': 32},
    'ExtraMsaStack': {
        'number_blocks': 4, 'msa_number_heads': 8, 'msa_head_embedding_dimension': 8,
        'msa_global_number_heads': 8, 'msa_global_head_embedding_dimension': 8,
        'pair_number_heads': 4, 'pair_head_embedding_dimension': 32,
        'intermediate_embedding': 32, 'msa_transition_channel_scaler': 4,
        'pair_stack_channel_scaler': 4, 'triangle_multiplication_embedding': 128
    },
    'EvoformerStack': {
        'number_blocks': 3, 'msa_number_heads': 8, 'msa_head_embedding_dimension': 32,
        'pair_number_heads': 4, 'pair_head_embedding_dimension': 32,
        'msa_transition_channel_scaler': 4, 'pair_stack_channel_scaler': 4,
        'intermediate_embedding': 32, 'triangle_multiplication_embedding': 128
    },
    'StructureModule': {
        'number_structure_module_iterations': 8, 'angle_representation_embedding': 128,
        'number_heads': 12, 'head_embedding_dimension': 16,
        'number_query_points': 4, 'number_value_points': 8,
        'number_torsion_angles': 7
    }
}
model = Model(configuration=configuration, device=device, dtype=dtype)
model.eval()
model.double()
with torch.no_grad():
    for param in model.parameters():
        param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

protein_id = "P90561"
model_input = ModelInput(structure_path=str(reference_folder / f"structures/{protein_id}.npz"),
                         msa_path=str(reference_folder / f"raw_msa/{protein_id}.a3m"),
                         record_path=str(reference_folder / f"records/{protein_id}.json"),
                         acceptance_slope_start=256, acceptance_slope_end=512,
                         residue_crop_size=None, emphasize_beginning_crops=False,
                         distribution_threshold=100, maximum_cluster_sequences=10,
                         maximum_extra_msa_sequences=11, mask_probability=0.0,
                         device=device, dtype=dtype)
data = model_input.get_data(number_samples=2, seed=42)

simple_input_dict = {
    'input_msa_feature': data['input_msa_feature'][:, :64, ...].unsqueeze(0),
    'input_extra_msa_feature': data['input_extra_msa_feature'][:, :64, ...].unsqueeze(0),
    'input_sequence_feature': data['input_sequence_feature'][:64, ...].unsqueeze(0),
    'input_residue_index_feature': data['input_residue_index_feature'][:64, ...].unsqueeze(0),
    'ground_truth_transformation_matrix': data['ground_truth_frames'][:64, ...].unsqueeze(0),
    'alternative_ground_truth_transformation_matrix': data['alternative_ground_truth_frames'][:64, ...].unsqueeze(0),
    'ground_truth_angles': data['ground_truth_angles'][:64, ...].unsqueeze(0),
    'alternative_ground_truth_angles': data['alternative_ground_truth_angles'][:64, ...].unsqueeze(0),
    'ground_truth_positions': data['ground_truth_global_positions'][:64, ...].unsqueeze(0),
    'alternative_ground_truth_positions': data['alternative_ground_truth_global_positions'][:64, ...].unsqueeze(0),
    'distogram_labels': data['distogram_labels'][:64, :64, ...].unsqueeze(0),
}
with torch.no_grad():
    simple_output = list(model(simple_input_dict).values())

out_names = ["model_angles", "model_frames", "model_final_positions",
             "model_position_mask", "model_pseudo_beta_positions",
             "model_overall_fape_loss", "model_auxillary_loss",
             "model_predicted_lddt_loss", "model_distogram_logits"]

print(f"Num outputs expected: {len(out_names)}, actual: {len(simple_output)}")

for i in range(len(out_names)):
    print(f"Expected name {out_names[i]}, shape from tensor: {simple_output[i].shape}")

# Let's see if the shapes match what the reference expects!
ref_logits = torch.load(reference_folder / "model_distogram_logits.pt", weights_only=True)
print("Reference logits shape:", ref_logits.shape)

