"""
File containing geometry utilities
 - Management and creation of quaternions
 - Management of rotational and translational matrices
 - Management of transformation application,e.t.c...
"""

import torch
from torch import nn


def create_3x3_rotation_matrix(ex, ey):
    """
    """

    ex = nn.functional.normalize(ex, dim=-1)
    ey = ey - ex * torch.sum(ex * ey, dim=-1, keepdim=True)
    ey = nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey)

    rotation_matrix = torch.stack(tensors=[ex, ey, ez], dim=-1)

    return rotation_matrix


def create_quaternion_from_axis_and_angle(phi, n):
    """"""

    # Normalise the vector if it hasn't been normalised
    n = nn.functional.normalize(n, dim=-1)

    quaternion_scaler_part = torch.cos(phi / 2).unsqueeze(dim=-1)
    quaternion_vector_part = torch.sin(phi / 2).unsqueeze(dim=-1) * n

    quaternion = torch.cat([quaternion_scaler_part, quaternion_vector_part], dim=-1)

    return quaternion


def multiply_quaternions(first_quaternion, second_quaternion):
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


def apply_quaternion_rotation_on_vector(quaternion, vector):
    padded_vector = nn.functional.pad(vector, pad=(1, 0), value=0)
    rotated_vector = multiply_quaternions(
        first_quaternion=multiply_quaternions(first_quaternion=quaternion, second_quaternion=padded_vector),
        second_quaternion=conjugate_quaternion(quaternion=quaternion))[..., 1:]

    return rotated_vector


def turn_quaternion_to_3x3_matrix(quaternion):
    batch_shape = quaternion.shape[:-1]

    eye = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    eye = eye.broadcast_to(batch_shape + (3, 3))

    x_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 0])
    y_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 1])
    z_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 2])

    rotation_matrix = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    return rotation_matrix


def assemble_4x4_transform_matrix(rotation_matrix, translation_vector):
    # pad arguments (left, right, top, bottom)
    padded_rotation_matrix = torch.nn.functional.pad(rotation_matrix, (0, 0, 0, 1), value=0)
    padded_translation_vector = torch.nn.functional.pad(translation_vector, (0, 1), value=1).unsqueeze(dim=-1)
    transformation_matrix = torch.cat(tensors=[padded_rotation_matrix, padded_translation_vector], dim=-1)

    return transformation_matrix


def apply_transformation_on_vector(transformation_matrix, vector):
    padded_vector = torch.nn.functional.pad(vector, (0, 1), value=1).unsqueeze(dim=-1)
    transformed_vector = torch.matmul(transformation_matrix, padded_vector).squeeze(dim=-1)[..., :-1]

    return transformed_vector


def create_4x4_transform_matrix(ex, ey, translation_vector):
    rotation_matrix = create_3x3_rotation_matrix(ex=ex, ey=ey)
    transformation_matrix = assemble_4x4_transform_matrix(rotation_matrix=rotation_matrix,
                                                          translation_vector=translation_vector)

    return transformation_matrix


def invert_4x4_transform_matrix(transformation_matrix):
    rotation_matrix = transformation_matrix[..., :3, :3]
    translation_matrix = transformation_matrix[..., :3, -1]

    inverted_rotation = torch.transpose(rotation_matrix, dim0=-2, dim1=-1)
    inverted_translation = -1 * torch.matmul(inverted_rotation, translation_matrix.unsqueeze(dim=-1)).squeeze(dim=-1)

    inverted_transformation_matrix = assemble_4x4_transform_matrix(
        rotation_matrix=inverted_rotation,
        translation_vector=inverted_translation)

    return inverted_transformation_matrix


def make_transformation_matrix_around_ex(phi):
    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype
    cos_phi, sin_phi = torch.unbind(phi, dim=-1)

    rotation_matrix = torch.zeros(batch_shape + (3, 3), device=device, dtype=dtype)
    rotation_matrix[..., 0, 0] = 1
    rotation_matrix[..., 1, 1] = cos_phi
    rotation_matrix[..., 2, 1] = sin_phi
    rotation_matrix[..., 1, 2] = -sin_phi
    rotation_matrix[..., 2, 2] = cos_phi

    translation_vector = torch.zeros(batch_shape + (3,), device=device, dtype=dtype)
    rotation_around_ex_transformation_matrix = assemble_4x4_transform_matrix(rotation_matrix=rotation_matrix,
                                                                             translation_vector=translation_vector)

    return rotation_around_ex_transformation_matrix
