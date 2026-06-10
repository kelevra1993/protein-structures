"""
File containing geometry utilities
 - Management and creation of quaternions
 - Management of rotational and translational matrices
 - Management of transformation application,e.t.c...
"""

import torch
from torch import nn
from utilities.constants import (rigid_group_atom_position_map, chi_angles_frame_centers, chi_angles_mask,
                                 atom_local_positions, atom_frame_indices, atom_mask, alternative_angle_mask,
                                 alternative_position_mask)
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

    vector_part = a1 * v2 + a2 * v1 + torch.linalg.cross(v1, v2, dim=-1)

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


def apply_transformation_on_vector(transformation_matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """
    Applies a 4x4 affine transformation matrix to a 3D vector.

    Converts the 3D vector into homogeneous coordinates (by appending a 1),
    applies the transformation, and then drops the homogeneous coordinate. This is 
    commonly used to transform local atom coordinates to the global frame.

    Args:
        transformation_matrix (torch.Tensor): The 4x4 affine transformation matrix.
            Expected shape: `(..., 4, 4)`.
        vector (torch.Tensor): The 3D vector to transform.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The transformed 3D vector.
            Shape: `(..., 3)`.
    """
    padded_vector = torch.nn.functional.pad(vector, (0, 1), value=1).unsqueeze(dim=-1)
    transformed_vector = torch.matmul(transformation_matrix, padded_vector).squeeze(dim=-1)[..., :-1]

    return transformed_vector


def create_4x4_transform_matrix(ex: torch.Tensor, ey: torch.Tensor, translation_vector: torch.Tensor) -> torch.Tensor:
    """
    Creates a 4x4 transformation matrix from two basis vectors and a translation vector.

    Used to define rigid body transformations from atomic coordinate references 
    (e.g., placing local side-chain frames onto the backbone).

    Args:
        ex (torch.Tensor): The first basis vector (x-axis).
            Expected shape: `(..., 3)`.
        ey (torch.Tensor): The second basis vector (y-axis).
            Expected shape: `(..., 3)`.
        translation_vector (torch.Tensor): The translation vector.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: The 4x4 transformation matrix.
            Shape: `(..., 4, 4)`.
    """
    rotation_matrix = create_3x3_rotation_matrix(ex=ex, ey=ey)
    transformation_matrix = assemble_4x4_transform_matrix(rotation_matrix=rotation_matrix,
                                                          translation_vector=translation_vector)

    return transformation_matrix


def invert_4x4_transform_matrix(transformation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Inverts a 4x4 affine transformation matrix.

    This takes advantage of the fact that the rotation part is orthogonal 
    (inverse is the transpose), making the inversion computationally efficient 
    compared to a general matrix inversion.

    Args:
        transformation_matrix (torch.Tensor): The 4x4 affine transformation matrix.
            Expected shape: `(..., 4, 4)`.

    Returns:
        torch.Tensor: The inverted 4x4 transformation matrix.
            Shape: `(..., 4, 4)`.
    """
    rotation_matrix = transformation_matrix[..., :3, :3]
    translation_matrix = transformation_matrix[..., :3, -1]

    inverted_rotation = torch.transpose(rotation_matrix, dim0=-2, dim1=-1)
    inverted_translation = -1 * torch.matmul(inverted_rotation, translation_matrix.unsqueeze(dim=-1)).squeeze(dim=-1)

    inverted_transformation_matrix = assemble_4x4_transform_matrix(
        rotation_matrix=inverted_rotation,
        translation_vector=inverted_translation)

    return inverted_transformation_matrix


def compute_dihedral_angle(point_1: torch.Tensor, point_2: torch.Tensor, point_3: torch.Tensor,
                           point_4: torch.Tensor) -> torch.Tensor:
    """
    Computes the dihedral (torsion) angle defined by four points in 3D space.

    The dihedral angle is the angle between the plane defined by (point_1, point_2, point_3)
    and the plane defined by (point_2, point_3, point_4), rotating around the central axis
    defined by the vector from point_2 to point_3.

    Args:
        point_1 (torch.Tensor): The first atomic coordinate.
            Expected shape: `(..., 3)`.
        point_2 (torch.Tensor): The second atomic coordinate (start of central axis).
            Expected shape: `(..., 3)`.
        point_3 (torch.Tensor): The third atomic coordinate (end of central axis).
            Expected shape: `(..., 3)`.
        point_4 (torch.Tensor): The fourth atomic coordinate.
            Expected shape: `(..., 3)`.

    Returns:
        torch.Tensor: A tensor containing the [cosine, sine] of the dihedral angle.
            Shape: `(..., 2)`.
    """
    vector_1 = point_2 - point_1
    vector_2 = point_3 - point_2
    vector_3 = point_4 - point_3

    # Calculate normal vectors to the two planes
    normal_to_plane_1 = torch.linalg.cross(vector_1, vector_2)
    normal_to_plane_2 = torch.linalg.cross(vector_2, vector_3)

    # Normalize the normal vectors
    normal_to_plane_1 = nn.functional.normalize(normal_to_plane_1, dim=-1)
    normal_to_plane_2 = nn.functional.normalize(normal_to_plane_2, dim=-1)

    # Cosine is the dot product of the normalized normals
    cos_angle = torch.sum(normal_to_plane_1 * normal_to_plane_2, dim=-1, keepdim=True)

    # Sine requires determining the direction of rotation.
    # We take the cross product of the normals and dot it with the normalized central axis.
    normalized_vector_2 = nn.functional.normalize(vector_2, dim=-1)
    cross_of_normals = torch.linalg.cross(normal_to_plane_1, normal_to_plane_2)
    sin_angle = torch.sum(cross_of_normals * normalized_vector_2, dim=-1, keepdim=True)

    return torch.cat([cos_angle, sin_angle], dim=-1)


def make_transformation_matrix_around_ex(phi: torch.Tensor) -> torch.Tensor:
    """
    Creates a 4x4 rotation matrix representing a rotation around the x-axis.

    In the AlphaFold II context, this is utilized to rotate around torsion angles 
    (which are parameterized by their cosine and sine) to construct the relative 
    transformations between adjacent rigid groups along the backbone and side-chains.

    Args:
        phi (torch.Tensor): The rotation angle provided in (cos_phi, sin_phi) format.
            Expected shape: `(..., 2)`.

    Returns:
        torch.Tensor: The 4x4 rotation matrix around the x-axis.
            Shape: `(..., 4, 4)`.
    """
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


def approximate_next_nitrogen(carbon_alpha: torch.Tensor, carbon: torch.Tensor, oxygen: torch.Tensor) -> torch.Tensor:
    """
    Approximates the position of the nitrogen atom of the next residue (N_{i+1})
    based on the idealized planar geometry of the peptide bond.

    Since the peptide bond is trigonal planar, CA_i, C_i, O_i, and N_{i+1} lie
    in the same plane. This function calculates the local position of N_{i+1} 
    in the standard coordinate space of the input atoms without converting them 
    to a temporary global plane.

    Idealized parameters used:
    - C-N bond length: ~1.329 Angstroms
    - CA-C-N angle: ~116.2 degrees

    Args:
        carbon_alpha (torch.Tensor): Coordinates of CA_i.
        carbon (torch.Tensor): Coordinates of C_i.
        oxygen (torch.Tensor): Coordinates of O_i.

    Returns:
        torch.Tensor: Approximated local coordinates of N_{i+1}.
    """
    device = carbon.device
    dtype = carbon.dtype

    # Create a local coordinate system at C_i
    # ex points towards CA_i
    # ey points roughly towards O_i (in the CA-C-O plane)
    ex = nn.functional.normalize(carbon_alpha - carbon, dim=-1)

    # ey is constructed to be orthogonal to ex, lying in the CA-C-O plane
    # pointing in the direction of oxygen
    vector_carbon_to_oxygen = oxygen - carbon
    ey = vector_carbon_to_oxygen - ex * torch.sum(ex * vector_carbon_to_oxygen, dim=-1, keepdim=True)
    ey = nn.functional.normalize(ey, dim=-1)

    # In this local plane, we want to place N_{i+1}.
    # N_{i+1} sits at a distance of 1.329 from C.
    # The angle CA-C-N is 116.2 degrees. Because O is on the +Y side and the geometry 
    # is trigonal planar (~120 deg apart), N must be on the -Y side.
    # Angle relative to ex (which points to CA) is -116.2 degrees.
    angle_offset = torch.deg2rad(torch.tensor(-116.2, device=device, dtype=dtype))
    bond_length = 1.329

    # Calculate the vector from C to N_{i+1}
    vector_carbon_to_next_nitrogen = bond_length * ((torch.cos(angle_offset) * ex) + (torch.sin(angle_offset) * ey))

    # Absolute position of approximated N_{i+1}
    nitrogen_next = carbon + vector_carbon_to_next_nitrogen

    return nitrogen_next



def compute_non_chi_transform_matrices() -> torch.Tensor:
    """
    Calculates the non-chi local frame transformations for all 21 residues (20 canonical + 1 unknown 'X').

    Constructs the transformations for the backbone, pre-omega, phi, and psi rigid groups. 
    These represent the base frames in the local coordinate systems before any torsion 
    angles are applied. 

    backbone_group: Identity
    pre_omega_group:
        ex: approximated N_{i+1} -> C
        ey: CA -> C
        t:  C
    phi_group:
        ex: CA -> N
        ey: CA -> C
        t:  N
    psi_group:
        ex: CA -> C
        ey: CA -> N
        t:  C

    Returns:
        torch.Tensor: The stacked transformation matrices for the non-chi rigid groups.
            Shape: `(21, 4, 4, 4)`, where the second dimension represents the 4 frames 
            (backbone, pre-omega, phi, psi).
    """

    non_chi_transforms = []

    for amino_acid, amino_acid_information in rigid_group_atom_position_map.items():
        backbone_transformation = torch.eye(4)

        # Approximate N_{i+1} for the pre-omega frame
        approximated_next_nitrogen = approximate_next_nitrogen(
            carbon_alpha=amino_acid_information["CA"],
            carbon=amino_acid_information["C"],
            oxygen=amino_acid_information["O"]
        )

        pre_omega_ex = approximated_next_nitrogen - amino_acid_information["C"]
        pre_omega_ey = amino_acid_information["CA"] - amino_acid_information["C"]

        pre_omega_translation = amino_acid_information["C"]
        
        pre_omega_transformation = create_4x4_transform_matrix(
            ex=pre_omega_ex,
            ey=pre_omega_ey,
            translation_vector=pre_omega_translation
        )

        phi_group_ex = amino_acid_information["N"] - amino_acid_information["CA"]
        # or even phi_group_ey = torch.tensor([1, 0, 0])
        phi_group_ey = amino_acid_information["C"] - amino_acid_information["CA"]

        phi_group_translation = amino_acid_information["N"]
        phi_group_transformation = create_4x4_transform_matrix(
            ex=phi_group_ex,
            ey=phi_group_ey,
            translation_vector=phi_group_translation)

        psi_group_ex = amino_acid_information["C"] - amino_acid_information["CA"]
        psi_group_ey = amino_acid_information["N"] - amino_acid_information["CA"]

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


def compute_chi_transform_matrices() -> torch.Tensor:
    """
    Calculates transforms for the local side-chain frames (chi1 to chi4) for
    all 21 residues (20 canonical + 1 unknown 'X').

    These frames track the side-chain atom positions. If the chi angles are not present
    for the given amino acid (according to `chi_angles_mask`), they are substituted by
    the Identity transform.

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
    (#SC0 - #SC3 denote the names of the side-chain atoms).

    Returns:
        torch.Tensor: Stacked transforms for the chi frames.
            Shape: `(21, 4, 4, 4)`. The second dim corresponds to the 4 chi frames.
    """

    # Note: For chi2, chi3 and chi4, ey is the inverse of the previous ex.
    # This means, that ey is (-1, 0, 0) in local coordinates for the frame.
    # Also note: For chi2, chi3, and chi4, ex starts at t of the previous transform.
    # This means, that the starting point is 0 in local coordinates.

    chi_transforms = torch.zeros((21, 4, 4, 4))
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


def compute_initial_rigid_transform_matrices() -> torch.Tensor:
    """
    Combines the non-chi and chi local frame transformations into a single tensor.

    This provides all 8 initial rigid group transformation frames (backbone, pre-omega,
    phi, psi, chi1, chi2, chi3, chi4) for the 21 residues.

    Returns:
        torch.Tensor: The stacked initial rigid transformations.
            Shape: `(21, 8, 4, 4)`.
    """

    rigid_transforms = torch.cat(tensors=[compute_non_chi_transform_matrices(),
                                          compute_chi_transform_matrices()], dim=1)

    return rigid_transforms


def compute_global_transform_matrices(transformation_matrix: torch.Tensor, residue_angles: torch.Tensor,
                                      sequence_amino_acid_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the global transformation matrices for all 8 rigid groups per residue.

    Applies the predicted backbone transformation and consecutive torsion rotations 
    to the initialized local frames, producing the final global transformation for 
    every rigid group in each residue of the sequence.

    Args:
        transformation_matrix (torch.Tensor): The global backbone transformations.
            Expected shape: `(..., number_residues, 4, 4)`.
        residue_angles (torch.Tensor): Torsion angles (omega, phi, psi, chi1, chi2, chi3, chi4) 
            provided as (cos, sin) pairs. Note: `omega` is not actively used in AF2.
            Expected shape: `(..., number_residues, 7, 2)`.
        sequence_amino_acid_labels (torch.Tensor): The amino acid types encoded as indices (0-19).
            Expected shape: `(..., number_residues)`.

    Returns:
        torch.Tensor: The global transform matrices for all rigid groups.
            Shape: `(..., number_residues, 8, 4, 4)`.
    """

    device = transformation_matrix.device
    dtype = transformation_matrix.dtype

    normalized_alpha = nn.functional.normalize(residue_angles, dim=-1)
    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(normalized_alpha, dim=-2)

    # Get global initialised frames (20,8,4,4)
    initial_rigid_transformations = compute_initial_rigid_transform_matrices().to(dtype=dtype, device=device)

    # Select the ones that are important for our sequence (..., number_residues, 8, 4, 4)
    initial_sequence_frames = initial_rigid_transformations[sequence_amino_acid_labels]

    # We will build the global frames list to avoid in-place modifications
    all_global_frames = []

    # 1. Backbone Frame (Frame 0)
    # The first frame is just the predicted backbone transformation
    backbone_frame = transformation_matrix
    all_global_frames.append(backbone_frame)

    # 2. Frames 1-4 (omega, phi, psi, chi1)
    # These are computed relative to the backbone frame (Frame 0)
    for i, angle in enumerate([omega, phi, psi, chi1], start=1):
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)
        
        # global = backbone * local_initial * rotation
        frame = torch.matmul(
            backbone_frame,
            torch.matmul(initial_sequence_frames[..., i, :, :], rotation_matrix)
        )
        all_global_frames.append(frame)

    # 3. Frames 5-7 (chi2, chi3, chi4)
    # These are computed hierarchically relative to the previous frame
    for i, angle in enumerate([chi2, chi3, chi4], start=5):
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)
        
        # global = previous_global * local_initial * rotation
        previous_frame = all_global_frames[i - 1]
        frame = torch.matmul(
            previous_frame,
            torch.matmul(initial_sequence_frames[..., i, :, :], rotation_matrix)
        )
        all_global_frames.append(frame)

    # Stack all frames along the rigid group dimension
    global_transform_matrices = torch.stack(all_global_frames, dim=-3)

    return global_transform_matrices


def compute_all_atom_coordinates(transformation_matrix: torch.Tensor, residue_angles: torch.Tensor,
                                 sequence_amino_acid_labels: torch.Tensor) -> tuple:
    """
    Calculates the 3D coordinates for all 37 possible atoms in each residue.

    By mapping local atom positions through the corresponding computed global 
    rigid group transformations, this outputs the final Cartesian coordinates 
    for every atom. It also returns a mask indicating which of the 37 atoms 
    are actually present for the given amino acid type.

    Args:
        transformation_matrix (torch.Tensor): The global backbone transformations.
            Expected shape: `(..., number_residues, 4, 4)`.
        residue_angles (torch.Tensor): Torsion angles (omega, phi, psi, chi1, chi2, chi3, chi4)
            provided as (cos, sin) pairs.
            Expected shape: `(..., number_residues, 7, 2)`.
        sequence_amino_acid_labels (torch.Tensor): The amino acid types encoded as indices (0-19).
            Expected shape: `(..., number_residues)`.

    Returns:
        tuple: A tuple containing three torch.Tensor objects:
            - global_positions: The global Cartesian coordinates for all 37 atoms.
                Shape: `(..., number_residues, 37, 3)`.
            - all_atom_mask: A boolean mask indicating the presence of each atom.
                Shape: `(..., number_residues, 37)`.
            - global_transformation_matrices: The intermediate global frame for every rigid group in every residue.
                Shape: `(..., number_residues, 37, 4, 4)`.
    """

    device = transformation_matrix.device
    dtype = transformation_matrix.dtype

    # (number_residues, 8, 4, 4)
    global_transformation_matrices = compute_global_transform_matrices(
        transformation_matrix=transformation_matrix,
        residue_angles=residue_angles,
        sequence_amino_acid_labels=sequence_amino_acid_labels)

    # (number_residues, 37, 3)
    local_positions = atom_local_positions[sequence_amino_acid_labels].to(device=device, dtype=dtype)

    # (number_residues, 37)
    frame_indices = atom_frame_indices[sequence_amino_acid_labels]

    # (number_residues, 37)
    all_atom_mask = atom_mask[sequence_amino_acid_labels]

    # First get the number of missing dimensions in order to properly gather indices
    dim_diff = global_transformation_matrices.ndim - frame_indices.ndim

    # We need to get frame_indices from (number_residues, 37) to (number_residues, 37, 4, 4)
    # Here we know it is 4x4, but we will use the dimension difference and then broadcast it.
    frame_indices = unsqueeze_tensor(frame_indices, number=dim_diff, direction="right")

    # (number_residues, 37, 4, 4)
    frame_indices = frame_indices.broadcast_to(
        frame_indices.shape[:-dim_diff] + global_transformation_matrices.shape[-dim_diff:])

    # (number_residues, 37, 4, 4)
    global_frames = torch.gather(global_transformation_matrices, dim=-3, index=frame_indices)

    # (number_residues, 37, 3)
    global_positions = apply_transformation_on_vector(transformation_matrix=global_frames,
                                                      vector=local_positions)

    return global_positions, all_atom_mask, global_transformation_matrices


def create_alternative_truth_transformation_matrix(transformation_matrix: torch.Tensor,
                                                   sequence_amino_acid_labels: torch.Tensor) -> torch.Tensor:
    """
    Creates alternative ground truth transformation matrices by accounting for 
    amino acid side-chain symmetries.

    Certain amino acids (e.g., Asp, Glu, Phe, Tyr) have symmetric side-chains 
    where a 180-degree rotation around a specific chi angle results in a 
    chemically identical structure. This function generates the alternative 
    set of transformations for all 8 rigid groups, which is typically used 
    during FAPE loss calculation to ensure the model isn't penalized for 
    predicting one of the two equivalent orientations.

    Args:
        transformation_matrix (torch.Tensor): The original ground truth global 
            transformation matrices for all 8 rigid groups.
            Expected shape: `(..., number_residues, 8, 4, 4)`.
        sequence_amino_acid_labels (torch.Tensor): The amino acid types encoded 
            as indices (0-19).
            Expected shape: `(..., number_residues)`.

    Returns:
        torch.Tensor: The alternative global transformation matrices.
            Shape: `(..., number_residues, 8, 4, 4)`.
    """

    device = transformation_matrix.device
    dtype = transformation_matrix.dtype

    batch_size, number_residues = sequence_amino_acid_labels.shape[:2]

    alternative_rotations = alternative_angle_mask.to(device)[sequence_amino_acid_labels]
    residue_angles = torch.tensor([1.0, 0.0]).repeat(batch_size, number_residues, 7, 1).to(device=device, dtype=dtype)

    # Apply rotation to get alternative rotations
    residue_angles = residue_angles * alternative_rotations

    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(residue_angles, dim=-2)

    alternative_transformation_matrix = transformation_matrix.clone()

    for transformation_index, angle in enumerate([omega, phi, psi, chi1], start=1):
        # Note we already normalised the angles therefore they are in (cos(phi), sin(phi)) format
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)

        # Just frame * rotational matrix.
        alternative_transformation_matrix[..., transformation_index, :, :] = torch.matmul(
            input=transformation_matrix[..., transformation_index, :, :],
            other=rotation_matrix
        )

    # Here we have to keep track of the previous transformation
    for transformation_index, angle in enumerate([chi2, chi3, chi4], start=5):
        rotation_matrix = make_transformation_matrix_around_ex(phi=angle)
        alternative_transformation_matrix[..., transformation_index, :, :] = torch.matmul(
            input=transformation_matrix[..., transformation_index, :, :],
            other=rotation_matrix
        )

    return alternative_transformation_matrix


def create_alternative_truth_positions(ground_truth_positions: torch.Tensor,
                                       sequence_amino_acid_labels: torch.Tensor) -> torch.Tensor:
    """
    Creates alternative ground truth atom positions by swapping the coordinates 
    of symmetric atoms.

    Certain amino acids have symmetric side-chains where a 180-degree rotation 
    results in a chemically identical structure. To prevent the model from being 
    penalized for predicting the alternative valid configuration, we generate 
    an alternative truth where these specific symmetric atoms are swapped:
    
    - Aspartate (ASP): Swaps OD1 <-> OD2
    - Glutamate (GLU): Swaps OE1 <-> OE2
    - Phenylalanine (PHE): Swaps CD1 <-> CD2 and CE1 <-> CE2
    - Tyrosine (TYR): Swaps CD1 <-> CD2 and CE1 <-> CE2
    
    This function uses `alternative_position_mask` which maps non-symmetric 
    atoms to themselves, and symmetric atoms to their mirrored counterpart.

    Args:
        ground_truth_positions (torch.Tensor): The original ground truth Cartesian coordinates.
            Expected shape: `(..., number_residues, 37, 3)`.
        sequence_amino_acid_labels (torch.Tensor): The amino acid types encoded as indices (0-19).
            Expected shape: `(..., number_residues)`.

    Returns:
        torch.Tensor: The alternative global positions with symmetric atoms swapped.
            Shape: `(..., number_residues, 37, 3)`.
    """
    # Get device
    device = sequence_amino_acid_labels.device

    # Shape becomes: (..., number_residues, 37)
    alternative_indices = alternative_position_mask.to(device)[sequence_amino_acid_labels].to(ground_truth_positions.device)

    # Expand the indices to cover the spatial dimension (x, y, z)
    # Shape becomes: (..., number_residues, 37, 3)
    alternative_indices_expanded = alternative_indices.unsqueeze(dim=-1).repeat(1, 1, 1, 3)

    # Gather to swap the coordinates in the atom dimension (dim=-2)
    alternative_ground_truth_positions = torch.gather(
        input=ground_truth_positions,
        dim=-2,
        index=alternative_indices_expanded
    )

    return alternative_ground_truth_positions


def create_alternative_truth_angles(ground_truth_angles: torch.Tensor,
                                    sequence_amino_acid_labels: torch.Tensor) -> torch.Tensor:
    """
    Creates alternative ground truth torsion angles by applying a 180-degree 
    rotation to symmetric side-chain angles.

    For specific symmetric amino acids (e.g., Aspartate, Glutamate, Phenylalanine, 
    Tyrosine), rotating a specific chi angle by 180 degrees (pi radians) results 
    in an identical structure. Since the angles are stored as (cos, sin) pairs, 
    a 180-degree rotation is equivalent to negating both values.

    This function utilizes `alternative_angle_mask`, which maps non-symmetric 
    angles to [1, 1] and symmetric angles to [-1, -1].

    Args:
        ground_truth_angles (torch.Tensor): The original ground truth torsion angles 
            as (cos, sin) pairs.
            Expected shape: `(..., number_residues, 7, 2)`.
        sequence_amino_acid_labels (torch.Tensor): The amino acid types encoded as indices (0-19).
            Expected shape: `(..., number_residues)`.

    Returns:
        torch.Tensor: The alternative torsion angles.
            Shape: `(..., number_residues, 7, 2)`.
    """
    # Get device
    device = sequence_amino_acid_labels.device

    # Retrieve the scaler mask using the sequence labels
    # Shape becomes: (..., number_residues, 7, 2)
    alternative_angle_scaler = alternative_angle_mask.to(device)[sequence_amino_acid_labels].to(ground_truth_angles.device)

    # Apply the scaler to the ground truth angles
    alternative_ground_truth_angles = ground_truth_angles * alternative_angle_scaler

    return alternative_ground_truth_angles
