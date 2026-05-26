"""
TODO : Note to self
Rename c_s -> single_representation_embedding
Rename c_z -> pair_representation_embedding
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
"""

import torch
import math

n_layer = 2
N = 3
c_m = 4
c_z = 5
c = 6
N_head = 7
N_seq = 8
N_extra = 9
N_res = 10
c_s = 11
n_qp = 12
n_pv = 13
n_torsion_angles = 7

feature_shapes = {
    'm': (N_seq, N_res, c_m),
    's': (N_res, c_s),
    'z': (N_res, N_res, c_z),
    'residue_index': (N_res,),
    'x': (N_res, 3),
    'q': (N_head, N_res, c),
    'k': (N_head, N_res, c),
    'v': (N_head, N_res, c),
    'qp': (N_head, n_qp, N_res, 3),
    'kp': (N_head, n_qp, N_res, 3),
    'vp': (N_head, n_pv, N_res, 3),
    'T': (N_res, 4, 4),
    'att_scores': (N_head, N_res, N_res),
    'a': (N_res, c),
    's_initial': (N_res, c_s),
    'alpha': (N_res, n_torsion_angles, 2),
    'F': (N_res,),
}

batched_feature_shapes = {
    key: (N,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2-i/5, 2+i/5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['F'] = torch.arange(test_inputs['F'].numel()).reshape(test_inputs['F'].shape) % 20