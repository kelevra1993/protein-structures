"""
File containing geometry utilities
 - Management and creation of quaternions
 - Management of rotational and translational matrices
 - Management of transformation application,e.t.c...
"""

import torch
from torch import nn
from utilities.constants import (rigid_group_atom_position_map, chi_angles_frame_centers, chi_angles_mask,
                                 atom_local_positions, atom_frame_indices, atom_mask)
from utilities.tensor_utilities import unsqueeze_tensor


def create_3x3_rotation_matrix(ex: torch.Tensor, ey: torch.Tensor) -> torch.Tensor:
    """
    Creates a 3x3 rotation matrix using two provided orthonormal base vectors.

    In the AlphaFold II context, this is often used to construct rigid group 
    frames from atom positions (e.g., creating the local frame for a side-chain 
    or backbone from coordinates). It uses Gram-Schmidt orthogonalization to 
    ensure `ey` is orthogonal to `ex`, and then computes the cross product for `ez`.

    Args:
        ex (torch.Tensor): The first basis vector, representing the x-axis.
            Expected shape: `(..., 3)`.
        ey (torch.Tensor): The second basis vector, representing the y-axis.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The resulting 3x3 rotation matrix.
            Shape: `(..., 3, 3)`.
    """

    ex = nn.functional.normalize(ex, dim=-1)
    ey = ey - ex * torch.sum(ex * ey, dim=-1, keepdim=True)
    ey = nn.functional.normalize(ey, dim=-1)
    ez = torch.linalg.cross(ex, ey)

    rotation_matrix = torch.stack(tensors=[ex, ey, ez], dim=-1)

    return rotation_matrix


def create_quaternion_from_axis_and_angle(phi: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    Creates a unit quaternion from a rotation angle and a rotation axis.

    Quaternions are used throughout the structure module to maintain valid 
    3D rotations without suffering from gimbal lock and to provide a numerically 
    stable representation for rigid body transformations during Invariant Point Attention.

    Args:
        phi (torch.Tensor): The rotation angle in radians.
            Expected shape: `(...)`.
        n (torch.Tensor): The axis of rotation as a 3D vector.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The resulting quaternion in (w, x, y, z) format.
            Shape: `(..., 4)`.
    """

    # Normalise the vector if it hasn't been normalised
    n = nn.functional.normalize(n, dim=-1)

    quaternion_scaler_part = torch.cos(phi / 2).unsqueeze(dim=-1)
    quaternion_vector_part = torch.sin(phi / 2).unsqueeze(dim=-1) * n

    quaternion = torch.cat([quaternion_scaler_part, quaternion_vector_part], dim=-1)

    return quaternion


def multiply_quaternions(first_quaternion: torch.Tensor, second_quaternion: torch.Tensor) -> torch.Tensor:
    """
    Performs quaternion multiplication to compose two rotations.

    This function is fundamental for combining successive rigid body rotations
    within the structure module. It expects quaternions in (w, x, y, z) format.

    Args:
        first_quaternion (torch.Tensor): The first quaternion.
            Expected shape: `(..., 4)`.
        second_quaternion (torch.Tensor): The second quaternion.
            Expected shape: `(..., 4)`.

    Returns:
        torch.Tensor: The product of the two quaternions.
            Shape: `(..., 4)`.
    """
    a1 = first_quaternion[..., 0:1]  # shape (*, 1)
    v1 = first_quaternion[..., 1:]  # shape (*, 3)

    a2 = second_quaternion[..., 0:1]  # shape (*, 1)
    v2 = second_quaternion[..., 1:]  # shape (*, 3)

    scalar_part = a1 * a2 - torch.sum(v1 * v2, dim=-1, keepdim=True)

    vector_part = a1 * v2 + a2 * v1 + torch.linalg.cross(v1, v2)

    quaternion_output = torch.cat(tensors=[scalar_part, vector_part], dim=-1)

    return quaternion_output


def conjugate_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Computes the conjugate of a quaternion.

    For unit quaternions, the conjugate represents the inverse rotation. 
    This is necessary when applying rotations to vectors or finding inverse transformations.

    Args:
        quaternion (torch.Tensor): The input quaternion in (w, x, y, z) format.
            Expected shape: `(..., 4)`.

    Returns:
        torch.Tensor: The conjugated quaternion.
            Shape: `(..., 4)`.
    """
    conjugated_quaternion = torch.cat(tensors=[quaternion[..., 0:1], - quaternion[..., 1:]], dim=-1)

    return conjugated_quaternion


def apply_quaternion_rotation_on_vector(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Applies a 3D rotation defined by a quaternion to a 3D vector.

    This operation computes `q * v * q_conjugate`, where the vector is treated 
    as a pure quaternion (w=0). This is used in the structure module to update 
    atom coordinates based on predicted rigid body transformations.

    Args:
        quaternion (torch.Tensor): The unit quaternion representing the rotation.
            Expected shape: `(..., 4)`.
        vector (torch.Tensor): The 3D vector to rotate.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The rotated 3D vector.
            Shape: `(..., 3)`.
    """
    padded_vector = nn.functional.pad(vector, pad=(1, 0), value=0)
    rotated_vector = multiply_quaternions(
        first_quaternion=multiply_quaternions(first_quaternion=quaternion, second_quaternion=padded_vector),
        second_quaternion=conjugate_quaternion(quaternion=quaternion))[..., 1:]

    return rotated_vector


def turn_quaternion_to_3x3_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion representation of a rotation into a 3x3 rotation matrix.

    This is necessary to interface between different parts of the AlphaFold II pipeline 
    that might require explicit matrix forms (e.g., assembling the final 4x4 transform matrices) 
    rather than the more stable quaternion form used for continuous updates.

    Args:
        quaternion (torch.Tensor): The quaternion to convert, in (w, x, y, z) format.
            Expected shape: `(..., 4)`.

    Returns:
        torch.Tensor: The equivalent 3x3 rotation matrix.
            Shape: `(..., 3, 3)`.
    """
    batch_shape = quaternion.shape[:-1]

    eye = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    eye = eye.broadcast_to(batch_shape + (3, 3))

    x_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 0])
    y_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 1])
    z_axis = apply_quaternion_rotation_on_vector(quaternion=quaternion, vector=eye[..., 2])

    rotation_matrix = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    return rotation_matrix


def assemble_4x4_transform_matrix(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor) -> torch.Tensor:
    """
    Assembles a 4x4 affine transformation matrix from a 3x3 rotation matrix 
    and a 3D translation vector.

    These 4x4 matrices represent the rigid groups (backbone and side-chains) 
    in homogeneous coordinates, which simplifies the application of transformations 
    to atom coordinates across the entire protein structure.

    Args:
        rotation_matrix (torch.Tensor): The 3x3 rotation component.
            Expected shape: `(..., 3, 3)`.
        translation_vector (torch.Tensor): The 3D translation component.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The assembled 4x4 transformation matrix.
            Shape: `(..., 4, 4)`.
    """
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


def compute_non_chi_transform_matrices():
    """
    Todo make it a little bit better documented
    backbone_group: Identity
    pre_omega_group: Identity
    phi_group:
        ex: CA -> N
        ey: (1, 0, 0) or CA -> C
        t:  N
    psi_group:
        ex: CA -> C
        ey: N  -> CA
        t:  C
    """

    non_chi_transforms = []

    for amino_acid, amino_acid_information in rigid_group_atom_position_map.items():
        backbone_transformation = torch.eye(4)

        # Todo Not really used, we could just consider removing it later
        pre_omega_transformation = torch.eye(4)

        phi_group_ex = amino_acid_information["N"] - amino_acid_information["CA"]
        # or even phi_group_ey = torch.tensor([1, 0, 0])
        phi_group_ey = amino_acid_information["C"] - amino_acid_information["CA"]

        phi_group_translation = amino_acid_information["N"]
        phi_group_transformation = create_4x4_transform_matrix(
            ex=phi_group_ex,
            ey=phi_group_ey,
            translation_vector=phi_group_translation)

        psi_group_ex = amino_acid_information["C"] - amino_acid_information["CA"]
        psi_group_ey = amino_acid_information["CA"] - amino_acid_information["N"]
        psi_group_translation = amino_acid_information["C"]
        psi_group_transformation = create_4x4_transform_matrix(
            ex=psi_group_ex,
            ey=psi_group_ey,
            translation_vector=psi_group_translation)

        amino_acid_non_chi_frames = torch.stack(tensors=[backbone_transformation,
                                                         pre_omega_transformation,
                                                         phi_group_transformation,
                                                         psi_group_transformation], dim=0)

        non_chi_transforms.append(amino_acid_non_chi_frames)

    non_chi_transforms = torch.stack(tensors=non_chi_transforms, dim=0)

    return non_chi_transforms


def compute_chi_transform_matrices():
    """
    Todo make it a little bit better documented
    Calculates transforms for the following local side-chain frames:
    chi1:
        ex: CA -> #SC0
        ey: CA -> N
        t:  #SC0
    chi2:
        ex: #SC0 -> #SC1
        ey: #SC0 -> CA
        t:  #SC1
    chi3:
        ex: #SC1 -> #SC2
        ey: #SC1 -> #SC0
        t: #SC2
    chi4:
        ex: #SC2 -> #SC3
        ey: #SC2 -> #SC1
        t: #SC3

    #SC0 - #SC3 denote the names of the side-chain atoms.
    If the chi angles are not present for the amino acid according to
    chi_angles_mask, they are substituted by the Identity transform.

    Returns:
        torch.tensor: Stacked transforms of shape (20, 4, 4, 4).
            The second dim corresponds to the different frames.
            The last two dims are the shape of the individual transforms.
    """

    # Note: For chi2, chi3 and chi4, ey is the inverse of the previous ex.
    # This means, that ey is (-1, 0, 0) in local coordinates for the frame.
    # Also note: For chi2, chi3, and chi4, ex starts at t of the previous transform.
    # This means, that the starting point is 0 in local coordinates.

    chi_transforms = torch.zeros((20, 4, 4, 4))
    for amino_acid_index, (amino_acid, amino_acid_information) in enumerate(rigid_group_atom_position_map.items()):

        side_chain_centers = chi_angles_frame_centers[amino_acid]

        for i in range(4):

            # No chi angle for this given amino acid residue so just use identity matrix
            if chi_angles_mask[amino_acid_index][i] == 0:
                chi_transforms[amino_acid_index, i] = torch.eye(4)
                continue

            center_atom = side_chain_centers[i]
            # Side chain matrix to be constructed
            ex = amino_acid_information[center_atom]

            if i == 0:
                ey = amino_acid_information["N"] - amino_acid_information["CA"]
            else:
                # we are actually always pointing backwards along ex axis from chi2 to chi4 for ey
                ey = torch.tensor([-1, 0, 0])

            transformation = create_4x4_transform_matrix(ex=ex,
                                                         ey=ey,
                                                         translation_vector=ex)
            chi_transforms[amino_acid_index, i] = transformation

    return chi_transforms


def compute_initial_rigid_transform_matrices():
    """
    todo improve documentation
    Calculates the non-chi transforms backbone_group, pre_omega_group, phi_group and psi_group,
    together with the chi transforms chi1, chi2, chi3, and chi4.

    Returns:
        torch.tensor: Transforms of shape (20, 8, 4, 4).
    """

    rigid_transforms = torch.cat(tensors=[compute_non_chi_transform_matrices(),
                                          compute_chi_transform_matrices()], dim=1)

    return rigid_transforms


def compute_global_transform_matrices(transformation_matrix, residue_angles, sequence_amino_acid_labels):
    """
    # todo very important to add shape of the output
    """

    device = transformation_matrix.device
    dtype = transformation_matrix.dtype

    normalized_alpha = nn.functional.normalize(residue_angles, dim=-1)
    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(normalized_alpha, dim=-2)

    # Get global initialised frames (20,8,4,4)
    initial_rigid_transformations = compute_initial_rigid_transform_matrices().to(dtype=dtype, device=device)

    # Select the ones that are important for our sequence (number_residues,8,4,4)
    global_transform_matrices = initial_rigid_transformations[sequence_amino_acid_labels]

    # This is just equivalent to multiplying by the identity matrix
    # Here we just the predicted backbone transformation matrix
    global_transform_matrices[..., 0, :, :] = transformation_matrix

    for transformation_index, angle in enumerate([omega, phi, psi, chi1], start=1):
        # Note we already normalised the angles therefore they are in (cos(phi), sin(phi)) format
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)

        # Just backbone * rotational matrix.
        # Note : We actually don't even need it for omega since omega is just junk for the model.
        global_transform_matrices[..., transformation_index, :, :] = torch.matmul(
            input=global_transform_matrices[..., 0, :, :],
            other=torch.matmul(
                input=global_transform_matrices[..., transformation_index, :, :],
                other=rotation_matrix
            ))

    # Here we have to keep track of the previous transformation
    for transformation_index, angle in enumerate([chi2, chi3, chi4], start=5):
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)
        global_transform_matrices[..., transformation_index, :, :] = torch.matmul(
            input=global_transform_matrices[..., transformation_index - 1, :, :],
            other=torch.matmul(
                input=global_transform_matrices[..., transformation_index, :, :],
                other=rotation_matrix
            ))

    return global_transform_matrices


def compute_all_atom_coordinates(transformation_matrix, residue_angles, sequence_amino_acid_labels):
    """
    todo to be better documented
    Args:
        T (torch.tensor): Global backbone transform for each amino acid. Shape (N_res, 4, 4).
        alpha (torch.tensor): Torsion angles for each amino acid. Shape (N_res, 7, 2).
            The angles are in the order (omega, phi, psi, chi1, chi2, chi3, chi4).
            Angles are given as (cos(a), sin(a)).
        F (torch.tensor): Label for each amino acid of shape (N_res,).
            Labels are encoded as 0: Ala, 1: Arg, ..., 19: Val.

    Returns:
        tuple: A tuple consisting of the following values:
            global_positions: Tensor of shape (N_res, 37, 3), containing the global positions
                for each atom for each amino acid.
            all_atom_mask: Boolean tensor of shape (N_res, 37), containing whether or not the atoms
                are present in the amino acids.
    """

    device = transformation_matrix.device
    dtype = transformation_matrix.dtype

    # (number_residues, 8, 4, 4)
    global_transforms = compute_global_transform_matrices(transformation_matrix=transformation_matrix,
                                                          residue_angles=residue_angles,
                                                          sequence_amino_acid_labels=sequence_amino_acid_labels)

    # (number_residues, 37, 3)
    local_positions = atom_local_positions[sequence_amino_acid_labels].to(device=device, dtype=dtype)

    # (number_residues, 37)
    frame_indices = atom_frame_indices[sequence_amino_acid_labels]

    # (number_residues, 37)
    all_atom_mask = atom_mask[sequence_amino_acid_labels]

    # First get the number of missing dimensions in order to properly gather indices
    dim_diff = global_transforms.ndim - frame_indices.ndim

    # We need to get frame_indices from (number_residues, 37) to (number_residues, 37, 4, 4)
    # Here we know it is 4x4, but we will use the dimension difference and then broadcast it.
    frame_indices = unsqueeze_tensor(frame_indices, number=dim_diff, direction="right")

    # (number_residues, 37, 4, 4)
    global_frame_indices = frame_indices.broadcast_to(
        frame_indices.shape[:-dim_diff] + global_transforms.shape[-dim_diff:])

    # (number_residues, 37, 3)
    global_positions = apply_transformation_on_vector(transformation_matrix=global_frame_indices,
                                                      vector=local_positions)

    return global_positions, all_atom_mask
