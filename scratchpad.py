import os.path

import torch
import numpy as np

from utilities.constants import alternative_position_mask, alternative_angle_mask, index_to_xxx
from utilities.geometry_utilities import create_alternative_truth_transformation_matrix
from utilities.tensor_utilities import get_device

np.set_printoptions(linewidth=500, threshold=np.inf)

from utilities.tensor_utilities import print_tensor_shape, print_tensor_list, specialised_one_hot_encoder
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss, \
    compute_local_distance_difference_test, compute_plddt_loss, rename_symmetric_ground_truth_metrics
import yaml
from utilities.data.dataloader import get_protein_dataloader

from tqdm import tqdm
import time



batch_size = 4
number_residues = 5
number_frames =8

amino_acid_residues = (torch.arange(batch_size * number_residues) % 20).reshape(batch_size, number_residues)

def create_all_atom_positions(batch_size: int, number_residues: int, flip: bool = False, random=False):
    if random:
        positions = torch.randperm(number_residues * 37 * 3).reshape(number_residues, 37, 3)
    else:
        positions = torch.arange(number_residues * 37 * 3).reshape(number_residues, 37, 3)

    if flip:
        positions = torch.flip(positions, dims=[-2])

    positions = positions.to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return positions
def create_transformation_matrices(batch_size: int,
                                   number_residues: int,
                                   number_frames: int = 8,
                                   delta=2):
    transformation_matrix = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, number_frames, 1, 1)

    # Replace the last column (equivalent to the translation with some known value > simple (x,x,x) translation)
    for k in range(number_residues):
        for j in range(number_frames):
            transformation_matrix[k, j, :3, -1] += k + j + 1 + delta

    # Accommodate batch
    transformation_matrix = transformation_matrix.repeat(batch_size, 1, 1, 1, 1)

    return transformation_matrix


# The position matrices
ground_truth_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues, flip=False)
predicted_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues, flip=True)
# Shape (number_residues, 37)
alternative_positions = alternative_position_mask[amino_acid_residues].unsqueeze(dim=-1).repeat(1, 1, 1, 3)
alternative_ground_truth_positions = torch.gather(ground_truth_positions, dim=2, index=alternative_positions)

# The transformation matrices
ground_truth_transformation_matrix = create_transformation_matrices(batch_size=batch_size,
                                                                    number_residues=number_residues,
                                                                    number_frames=number_frames, delta=0)
predicted_transformation_matrix = create_transformation_matrices(batch_size=batch_size,
                                                                 number_residues=number_residues,
                                                                 number_frames=number_frames, delta=1)

# Swap change the rotations of the transformation matrices
alternative_rotations = alternative_angle_mask[amino_acid_residues]
alternative_rotations = alternative_rotations.repeat(batch_size, 1, 1, 1)
residue_angles = torch.tensor([1.0, 0.0]).repeat(batch_size, number_residues, 7, 1)
alternative_ground_truth_transformation_matrix = create_alternative_truth_transformation_matrix(
    transformation_matrix=ground_truth_transformation_matrix,
    sequence_amino_acid_labels=amino_acid_residues)


print(ground_truth_transformation_matrix.shape)
exit()
for batch_index in range(batch_size):
    modified_ground_truth_positions, modified_ground_truth_transformation_matrix = rename_symmetric_ground_truth_metrics(
        predicted_positions=predicted_positions[batch_index],
        ground_truth_transformation_matrix=ground_truth_transformation_matrix[batch_index],
        ground_truth_positions=ground_truth_positions[batch_index],
        alternative_ground_truth_transformation_matrix=alternative_ground_truth_transformation_matrix[batch_index],
        alternative_ground_truth_positions=alternative_ground_truth_positions[batch_index],
        sequence_amino_acid_labels=amino_acid_residues[batch_index],
    )

for b in range(batch_size):
    for r in range(number_residues):
        aa_r = r % 20
        if index_to_xxx[aa_r] not in ["ASP", "GLU", "TYR", "PHE"]:
            continue
        for n in range(37):
            gt_atom = ground_truth_positions[b, aa_r, n].numpy()
            alt_atom = alternative_ground_truth_positions[b, aa_r, n].numpy()

            if sum(alt_atom - gt_atom) != 0:
                print(f"Amino Acid Residue {index_to_xxx[aa_r]} For Atom {n:02}")
                print(15 * '-')
                print(gt_atom, alt_atom)
                print(30 * '-')
exit()
exit()
# Testing the new PyTorch DataLoaders
print("--- Benchmarking PyTorch DataLoaders ---")

data_folder = "/home/robert_kelevra/Data/protein_data/openfold_data"
train_split = "dataset_splits/Train.json"
validation_split = "dataset_splits/Validation.json"

number_epochs = 2
number_steps_per_epoch = 100  # Limit steps for quick benchmarking

# Load Configurations
config_path = "configurations/template_configuration.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

train_config = config["TrainDataConfiguration"]

# Temporarily override for quick benchmarking without massive MSAs
train_config["maximum_cluster_sequences"] = 32
train_config["maximum_extra_msa_sequences"] = 64
train_config["residue_crop_size"] = None
train_config["number_recycle_cycles"] = 10

train_loader = get_protein_dataloader(
    data_folder=data_folder,
    split_file_path=train_split,
    batch_size=1,
    shuffle=False,
    num_workers=10,
    **train_config)

for epoch in range(number_epochs):
    start_time = time.time()
    print(f"\nEpoch {epoch + 1}/{number_epochs}")

    for batch_index, batch_data in tqdm(enumerate(train_loader), total=number_steps_per_epoch, desc="Loading Batches"):
        if batch_index >= number_steps_per_epoch:
            break
        print(f"  Input Sequence Feature Shape: {batch_data['input_sequence_feature'].shape}")
        print(f"  Sequence Labels Shape: {batch_data['sequence_labels'].shape}")
        print(f"  Input MSA Feature Shape: {batch_data['input_msa_feature'].shape}")
        print(f"  Input Extra MSA Feature Shape: {batch_data['input_extra_msa_feature'].shape}")
        print(f"  Input Residue Index Feature Shape: {batch_data['input_residue_index_feature'].shape}")
        print(f"  Ground Truth Global Positions Shape: {batch_data['ground_truth_global_positions'].shape}")
        print(f"  Ground Truth Local Positions Shape: {batch_data['ground_truth_local_positions'].shape}")
        print(f"  Ground Truth Frames Shape: {batch_data['ground_truth_frames'].shape}")
        print(f"  Ground Truth Angles Shape: {batch_data['ground_truth_angles'].shape}")
        exit()

    end_time = time.time()
    epoch_duration = end_time - start_time
    samples_per_second = number_steps_per_epoch / epoch_duration
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s ({samples_per_second:.2f} samples/s)")

exit()

batch_size = 2
number_residues = 5

pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)

for k in range(number_residues):
    pred_tr_m[k, :3, -1] += k + 1
    gt_tr_m[k, :3, -1] += k + 2

pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))

pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
    (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)

gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
    (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,
                                                                                                        1)

print(pred_tr_m.shape)
print(gt_tr_m.shape)
exit()
fape_loss = compute_fape_loss(predicted_transformation_matrix=pred_tr_m,
                              predicted_positions=pred_carbon_alpha_positions,
                              ground_truth_transformation_matrix=gt_tr_m,
                              ground_truth_positions=gt_carbon_alpha_positions,
                              length_scaler=2,
                              epsilon=2e-4,
                              distance_clamp=2.0)

exit()


def create_all_atom_positions(batch_size: int, number_residues: int, flip: bool = False, random=False):
    if random:
        positions = torch.randperm(number_residues * 37 * 3).reshape(number_residues, 37, 3)
    else:
        positions = torch.arange(number_residues * 37 * 3).reshape(number_residues, 37, 3)

    if flip:
        positions = torch.flip(positions, dims=[-2])

    positions = positions.to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return positions


batch_size = 8
number_residues = 10
dtype = torch.float32
scale = 30
threshold = 15
distance_thresholds = [9.5, 1.0, 2.0, 4.0]

# Get the residues of the sequence
amino_acid_residues = (torch.arange(batch_size * number_residues) % 20).reshape(batch_size, number_residues)
amino_acid_residues = torch.flip(amino_acid_residues, dims=[-1])

# Create the positions of predictions and ground truths
# (batch, number_residues, 37, 3)
prediction_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues,
                                                 random=True) / scale
ground_truth_positions = create_all_atom_positions(batch_size=batch_size, number_residues=number_residues,
                                                   flip=True) / (scale / 2)

local_difference_distance_test = compute_local_distance_difference_test(
    # prediction_positions=ground_truth_positions,
    prediction_positions=prediction_positions,
    ground_truth_positions=ground_truth_positions,
    distance_thresholds=distance_thresholds)

from architecture_modules.lddt_module.lddt_module import LddtModule

# Create single_representation
embedding_dimension = 100
single_representation = torch.randperm(batch_size * number_residues * embedding_dimension, dtype=torch.float64).reshape(
    batch_size,
    number_residues,
    embedding_dimension)

lddt_module = LddtModule(single_representation_embedding=embedding_dimension,
                         intermediate_embedding=int(embedding_dimension / 4),
                         device=torch.device("cpu"),
                         dtype=torch.float64)

lddt_logits, predicted_lddt_probabilities, plddt = lddt_module(single_representation=single_representation)

plddt_loss = compute_plddt_loss(
    ground_truth_lddt=local_difference_distance_test,
    predicted_lddt_logits=lddt_logits,
    lddt_bins=lddt_module.lddt_bins
)

# print(residue_angles)
# print_tensor_shape(tensor=alternative_rotations)
# print_tensor_shape(tensor=residue_angles)
# residue_angles = residue_angles*alternative_rotations
# print(residue_angles)
# print(alternative_rotations)


# for b in range(batch_size):
#     for r in range(number_residues):
#         aa_r = r % 20
#         for f in range(number_frames):
#             print(f"Amino Acid Residue {index_to_xxx[aa_r]} For Angle Frame {f:02}")
#             print(ground_truth_transformation_matrix[b, aa_r, f].numpy())
#             print(15 * '-')
#             print(alternative_ground_truth_transformation_matrix[b, aa_r, f].numpy())
#             print(30 * '-')
# exit()
# alternative_ground_truth_transformation_matrix =


# scaler = 3
# pred_gt_delta = 2
# # Predicted angles number_residues, 7, 2
# predicted_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
# predicted_y = (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
# predicted_unnormalised_torsion_angles = torch.stack(tensors=[predicted_x, predicted_y], dim=-1)
#
# # Ground Truth angles number_residues, 7, 2
# gt_x = (torch.arange(7) + 1).repeat(number_residues, 1).to(dtype=dtype)
# gt_y = pred_gt_delta * (torch.arange(7) + scaler).repeat(number_residues, 1).to(dtype=dtype)
# ground_truth_torsion_angles = torch.nn.functional.normalize(torch.stack(tensors=[gt_x, gt_y], dim=-1), dim=-1)
#
#
#

#
#
# # Get the alternative truths for the torsion angles
# alternative_scaler = alternative_angle_mask[amino_acid_residues]
# # print(f"{angle_alternative_truth_mask.shape=}")
# # print(f"{residues_to_change.shape=}")
# # print(alternative_scaler)
# alternative_ground_truth_torsion_angles = alternative_scaler * ground_truth_torsion_angles
# # print(ground_truth_torsion_angles.numpy())
# # print(20*'--')
# # print(alternative_ground_truth_torsion_angles.numpy())
#
# # Add batch size to the different tensors
# predicted_unnormalised_torsion_angles = predicted_unnormalised_torsion_angles.repeat(batch_size, 1, 1, 1)
# ground_truth_torsion_angles = ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)
# alternative_ground_truth_torsion_angles = alternative_ground_truth_torsion_angles.repeat(batch_size, 1, 1, 1)
#
#
# auxillary_loss = compute_torsion_angle_loss(
#     predicted_unnormalised_angles=predicted_unnormalised_torsion_angles,
#     ground_truth_angles=ground_truth_torsion_angles,
#     alternative_ground_truth_angles=alternative_ground_truth_torsion_angles,
#     angle_norm_loss_scaler=0.2)
# print(auxillary_loss.numpy())
# # pred_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
# # gt_tr_m = torch.eye(4).to(torch.float64).unsqueeze(dim=0).repeat(number_residues, 1, 1)
# #
# # for k in range(number_residues):
# #     pred_tr_m[k, :3, -1] += k + 1
# #     gt_tr_m[k, :3, -1] += k + 2
# #
# # pred_tr_m = pred_tr_m.repeat((batch_size, 1, 1, 1))
# # gt_tr_m = gt_tr_m.repeat((batch_size, 1, 1, 1))
# #
# # pred_carbon_alpha_positions = torch.ones(number_residues, 3) * (
# #     (torch.arange(1, number_residues + 1, 1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)
# #
# # gt_carbon_alpha_positions = torch.ones(number_residues, 3) * (
# #     (torch.arange(number_residues + 1, 1, -1)).unsqueeze(dim=-1)).to(torch.float64).unsqueeze(0).repeat(batch_size, 1,1)
