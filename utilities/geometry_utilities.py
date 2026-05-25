"""
File containing geometry utilities
 - Management and creation of quaternions
 - Management of rotational and translational matrices
 - Management of transformation application,e.t.c...
"""

import torch
from torch import nn


def create_3x3_rotational_matrix(ex, ey):
    """
    """


    ex = nn.functional.normalize(ex, dim=-1)
    ey = ey - ex * torch.sum(ex * ey, dim=-1, keepdim=True)
    ey = nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey)

    rotational_matrix = torch.stack(tensors=[ex, ey, ez], dim=-1)


    return rotational_matrix


def create_quaternion_from_axis_and_angle(phi, n):
    """"""

    # Normalise the vector if it hasn't been normalied
    n = nn.functional.normalize(n, dim=-1)

    quaternion_scaler_part = torch.cos(phi / 2).unsqueeze(dim=-1)
    quaternion_vector_part = torch.sin(phi / 2).unsqueeze(dim=-1) * n

    quaternion = torch.cat([quaternion_scaler_part, quaternion_vector_part], dim=-1)


    return quaternion

def quaternion_multiplication(first_quaternion, second_quaternion):


    a1 = first_quaternion[..., 0:1]  # shape (*, 1)
    v1 = first_quaternion[..., 1:]  # shape (*, 3)

    a2 = second_quaternion[..., 0:1]  # shape (*, 1)
    v2 = second_quaternion[..., 1:]  # shape (*, 3)

    scalar_part = a1 * a2 - torch.sum(v1 * v2, dim=-1, keepdim=True)

    vector_part = a1 * v2 + a2 * v1 + torch.linalg.cross(v1, v2)

    quaternion_output = torch.cat(tensors=[scalar_part, vector_part], dim=-1)

    return quaternion_output

def conjugate_quaternion(quaternion):


    conjugated_quaternion = torch.cat(tensors=[quaternion[..., 0:1], - quaternion[..., 1:]], dim=-1)

    return conjugated_quaternion