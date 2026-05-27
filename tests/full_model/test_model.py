"""
TODO : Note to self for variables to be renamed
Rename c_s -> single_representation_embedding
Rename c_z -> pair_representation_embedding
Rename c_m -> msa_embedding
Rename c_e -> extra_msa_embedding
Rename tf_dim -> input_sequence_feature_dimension (generally 21)
Rename f_e -> input_extra_msa_feature_dimension (generally 25)
Rename num_blocks_extra_msa -> number_extra_msa_blocks
Rename num_blocks_evoformer -> number_evoformer_blocks
Rename n_query_points -> number_query_points
Rename n_point_values -> number_value_points
Rename N_head -> number_heads
Rename c -> head_embedding_dimension
Rename s -> single_representation
Rename z -> pair_representation
Rename q -> query_tensor
Rename k -> key_tensor
Rename qp-> query_point_tensor
Rename kp-> key_point_tensor
Rename T -> transformation_matrix
Rename warp_3d_point -> apply_transformation_on_vector
Rename N_res -> number_residues
Rename N_cycle -> number_cycles
Rename prev_m -> previous_msa_representation_tensor
Rename prev_z -> previous_pair_representation_tensor
Rename prev_pseudo_beta_x -> previous_pseudo_carbon_beta_positions
Rename N_seq -> number_clusters
Rename N_extra -> number_extra_sequences
Rename msa_feat_dim -> msa_feature_dimension
Rename F -> sequence_amino_acid_labels
Rename N_layers -> num_layers
Rename n_torsion_angles -> number_torsion_angles
"""


import math
import torch
from torch import nn

N_cycle = 2
num_blocks_evoformer = 3
num_blocks_extra_msa = 4
N = 5
c_m = 6
c_z = 7
c = 8
N_head = 9
N_seq = 10
N_extra = 11
N_res = 12
c_s = 384
n_qp = 14
n_pv = 15
n_torsion_angles = 7
tf_dim = 21
msa_feat_dim = 49
f_e = 18
c_e = 19

feature_shapes = {
    'msa_feat': (N_seq, N_res, msa_feat_dim, N_cycle),
    'target_feat': (N_res, tf_dim, N_cycle),
    'residue_index': (N_res,N_cycle),
    'extra_msa_feat': (N_extra, N_res, f_e, N_cycle),
}

batched_feature_shapes = {
    key: (N,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2-i/5, 2+i/5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['residue_index'] = torch.arange(N_res).view(N_res, 1).broadcast_to(N_res, N_cycle)
test_inputs['target_feat'] = nn.functional.one_hot(torch.arange(N_res)%20, num_classes=tf_dim).double()
test_inputs['target_feat'] = test_inputs['target_feat'].unsqueeze(-1).broadcast_to(feature_shapes['target_feat'])

test_inputs['batch'] = {
    'msa_feat': test_inputs['msa_feat'],
    'target_feat': test_inputs['target_feat'],
    'residue_index': test_inputs['residue_index'],
    'extra_msa_feat': test_inputs['extra_msa_feat'],
}