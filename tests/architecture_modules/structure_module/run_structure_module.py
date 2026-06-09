import os
import torch
from pathlib import Path

from architecture_modules.structure_module.structure_module import StructureModule
from tests.utilities.testing_utilities import check_nn_module_method, get_structure_module_test_inputs
from utilities.data.input import ModelInput


def run_structure_module_forward():
    device = torch.device('cpu')
    dtype = torch.float64

    config, combined_inputs = get_structure_module_test_inputs()

    reference_folder = os.path.join(os.path.dirname(__file__), 'reference_values')

    simple_inputs = {k: v[0] for k, v in combined_inputs.items()}
    batched_inputs = {k: v[1] for k, v in combined_inputs.items()}

    # additional config settings
    config["number_iterations"] = 8
    target_number_residues = config["number_residues"]

    # Data extraction from ModelInput
    # todo the testing data will need to be moved later to the adeaquate folders.
    project_root = Path(__file__).parents[3]
    protein_id = "P90561"
    structure_path = project_root / f"data_examples/openfold/structures/{protein_id}.npz"
    msa_path = project_root / f"data_examples/openfold/raw_msa/{protein_id}.a3m"
    record_path = project_root / f"data_examples/openfold/records/{protein_id}.json"

    model_input = ModelInput(structure_path=str(structure_path), msa_path=str(msa_path), record_path=str(record_path),
                             acceptance_slope_start=256, acceptance_slope_end=512,
                             residue_crop_size=None, emphasize_beginning_crops=False,
                             distribution_threshold=100,
                             maximum_cluster_sequences=32, maximum_extra_msa_sequences=64, mask_probability=0.0,
                             device=device, dtype=dtype)

    data = model_input.get_data(number_samples=config["number_iterations"], seed=42)

    # Initialize the module
    module = StructureModule(
        single_representation_embedding=config["single_representation_embedding"],
        pair_representation_embedding=config["pair_representation_embedding"],
        device=device,
        dtype=dtype,
        number_iterations=config["number_iterations"],
        number_torsion_angles=7,
        angle_representation_embedding=20,
        number_query_points=4,  # 4
        number_value_points=8,  # 8
        number_heads=12,  # 12
        head_embedding_dimension=16,  # 16
    )

    # setup ground truth data
    frames = data["ground_truth_frames"][:target_number_residues, :, :, :, 0]
    alternative_frames = data["alternative_ground_truth_frames"][:target_number_residues, :, :, :, 0]
    angles = data["ground_truth_angles"][:target_number_residues, :, :, 0]
    alternative_angles = data["alternative_ground_truth_angles"][:target_number_residues, :, :, 0]
    positions = data["ground_truth_global_positions"][:target_number_residues, :, :, 0]
    alternative_positions = data["alternative_ground_truth_global_positions"][:target_number_residues, :, :, 0]
    sequence_labels = data["sequence_labels"][:target_number_residues, 0]


    input_tensor_dictionary = {
        "single_representation": simple_inputs["single_representation"].to(device),
        "pair_representation": simple_inputs["pair_representation"].to(device),
        "sequence_amino_acid_labels": sequence_labels,
        "ground_truth_transformation_matrix": frames,
        "alternative_ground_truth_transformation_matrix": alternative_frames,
        "ground_truth_angles": angles,
        "alternative_ground_truth_angles": alternative_angles,
        "ground_truth_positions": positions,
        "alternative_ground_truth_positions": alternative_positions}

    batched_input_tensor_dictionary = {
        "single_representation": batched_inputs["single_representation"].to(device),
        "pair_representation": batched_inputs["pair_representation"].to(device),
        "sequence_amino_acid_labels": sequence_labels.unsqueeze(0).repeat(config["batch_size"], 1),
        "ground_truth_transformation_matrix": frames.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1, 1),
        "alternative_ground_truth_transformation_matrix": alternative_frames.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1, 1),
        "ground_truth_angles": angles.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1),
        "alternative_ground_truth_angles": alternative_angles.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1),
        "ground_truth_positions": positions.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1),
        "alternative_ground_truth_positions": alternative_positions.unsqueeze(0).repeat(config["batch_size"], 1, 1, 1),}

    output_tensor_names = [
        "structure_module_angles",
        "structure_module_frames",
        "structure_module_final_positions",
        "structure_module_position_mask",
        "structure_module_pseudo_beta_positions"]

    check_nn_module_method(
        module=module,
        input_tensor_dictionary=input_tensor_dictionary,
        output_tensor_names=output_tensor_names,
        reference_folder=reference_folder,
        batch_size=config["batch_size"],
        batched_input_tensor_dictionary=batched_input_tensor_dictionary
    )
    print(" - StructureModule Test Completed Successfuly.")


run_structure_module_forward()
