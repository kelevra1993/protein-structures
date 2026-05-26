import torch
from torch import nn
from typing import Tuple

from architecture_modules.structure_module.invariant_point_attention_module import InvariantPointAttention
from utilities.geometry_utilities import compute_all_atom_coordinates, assemble_4x4_transform_matrix, \
    turn_quaternion_to_3x3_matrix

# Here we will have to design this explicitly
from utilities.constants import atom_types, canonical_amino_acid_residues


class StructureModuleTransition(nn.Module):
    """
    A transition layer within the Structure Module that updates the single representation
    through a series of linear transformations and non-linearities.

    This module is used in each iteration of the StructureModule after the Invariant Point
    Attention (IPA) layer to refine the residue-level features before they are used
    to predict backbone updates and torsion angles.
    """

    def __init__(self, single_representation_embedding: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the StructureModuleTransition module.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
        """
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.device = device
        self.dtype = dtype

        self.first_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                        out_features=self.single_representation_embedding,
                                        device=self.device, dtype=self.dtype)
        self.second_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                         out_features=self.single_representation_embedding,
                                         device=self.device, dtype=self.dtype)
        self.third_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                        out_features=self.single_representation_embedding,
                                        device=self.device, dtype=self.dtype)
        self.single_representation_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.single_representation_embedding,
            device=self.device, dtype=self.dtype)
        self.ReLu = nn.ReLU()

    def forward(self, single_representation: torch.Tensor) -> torch.Tensor:
        """
        Applies the transition layers to the single representation.

        Args:
            single_representation (torch.Tensor): Input residue-level features.
                Shape: `(..., number_residues, single_representation_embedding)`.

        Returns:
            torch.Tensor: Updated single representation.
                Shape: `(..., number_residues, single_representation_embedding)`.
        """
        single_representation_output = (single_representation +
                                        self.third_embedder(self.ReLu(
                                            self.second_embedder(self.ReLu(
                                                self.first_embedder(single_representation))))))
        single_representation_output = self.single_representation_layer_normalizer(single_representation_output)

        return single_representation_output


class BackboneUpdate(nn.Module):
    """
    Predicts updates to the backbone transformation matrices from the single representation.

    In each iteration of the StructureModule, this module takes the refined single
    representation and predicts a 6D vector for each residue (3 for an unnormalized
    quaternion and 3 for a translation vector). These are then used to update the
    local 4x4 transformation matrices (frames) for each residue.
    """

    def __init__(self, single_representation_embedding: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the BackboneUpdate module.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation
                (`single_representation_dimension`).
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
        """
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.device = device
        self.dtype = dtype
        self.backbone_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                           out_features=6, device=self.device, dtype=self.dtype)

    def forward(self, single_representation: torch.Tensor) -> torch.Tensor:
        """
        Computes 4x4 transformation updates from the single representation.

        Args:
            single_representation (torch.Tensor): Input residue-level features.
                Shape: `(..., number_residues, single_representation_dimension)`.

        Returns:
            torch.Tensor: Local transformation matrices representing backbone updates.
                Shape: `(..., number_residues, 4, 4)`.
        """
        output = self.backbone_embedder(single_representation)
        un_normalised_quaternion, translation_vector = output[..., :3], output[..., 3:]

        # Quaternion looks like this (1, x, y, z) and then normalised to represent a rotation
        quaternion = torch.nn.functional.pad(un_normalised_quaternion, (1, 0), value=1.0)
        quaternion = torch.nn.functional.normalize(quaternion, dim=-1)

        transformation_matrix = assemble_4x4_transform_matrix(
            rotation_matrix=turn_quaternion_to_3x3_matrix(quaternion=quaternion),
            translation_vector=translation_vector)

        return transformation_matrix


class AngleResNetLayer(nn.Module):
    """
    A residual layer used within the AngleResNet for torsion angle prediction.
    """

    def __init__(self, angle_representation_embedding: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the AngleResNetLayer.

        Args:
            angle_representation_embedding (int): Feature dimension of the angle representation.
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
        """
        super().__init__()
        self.angle_representation_embedding = angle_representation_embedding
        self.device = device
        self.dtype = dtype

        self.first_angle_embedder = nn.Linear(in_features=self.angle_representation_embedding,
                                              out_features=self.angle_representation_embedding,
                                              device=self.device, dtype=self.dtype)

        self.second_angle_embedder = nn.Linear(in_features=self.angle_representation_embedding,
                                               out_features=self.angle_representation_embedding,
                                               device=self.device, dtype=self.dtype)
        self.relu = nn.ReLU()

    def forward(self, angle_representation: torch.Tensor) -> torch.Tensor:
        """
        Applies residual layers to the angle representation.

        Args:
            angle_representation (torch.Tensor): Input angle features.
                Shape: `(..., angle_representation_embedding)`.

        Returns:
            torch.Tensor: Updated angle representation.
                Shape: `(..., angle_representation_embedding)`.
        """
        angle_representation = angle_representation + self.second_angle_embedder(
            self.relu(self.first_angle_embedder(self.relu(angle_representation))))

        return angle_representation


class AngleResNet(nn.Module):
    """
    Predicts protein torsion angles from the single representation using a residual network.
    """

    def __init__(self, single_representation_embedding: int, angle_representation_embedding: int,
                 device: torch.device, dtype: torch.dtype, number_torsion_angles: int = 7):
        """
        Initializes the AngleResNet module.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            angle_representation_embedding (int): Hidden dimension for angle prediction.
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
            number_torsion_angles (int): Number of torsion angles to predict. Defaults to 7.
        """
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.angle_representation_embedding = angle_representation_embedding
        self.number_torsion_angles = number_torsion_angles
        self.device = device
        self.dtype = dtype
        self.initial_single_representation_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                                                out_features=self.angle_representation_embedding,
                                                                device=self.device, dtype=self.dtype)
        self.current_single_representation_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                                                out_features=self.angle_representation_embedding,
                                                                device=self.device, dtype=self.dtype)

        self.angle_resnet_layers = nn.ModuleList([
            AngleResNetLayer(angle_representation_embedding=self.angle_representation_embedding,
                             device=self.device, dtype=self.dtype),
            AngleResNetLayer(angle_representation_embedding=self.angle_representation_embedding,
                             device=self.device, dtype=self.dtype)])

        self.torsion_angles_output_embedder = nn.Linear(in_features=self.angle_representation_embedding,
                                                        out_features=2 * self.number_torsion_angles,
                                                        device=self.device, dtype=self.dtype)
        self.relu = nn.ReLU()

    def forward(self, single_representation: torch.Tensor, initial_single_representation: torch.Tensor) -> torch.Tensor:
        """
        Predicts torsion angles.

        Args:
            single_representation (torch.Tensor): Current single representation.
                Shape: `(..., number_residues, single_representation_embedding)`.
            initial_single_representation (torch.Tensor): Initial single representation features.
                Shape: `(..., number_residues, single_representation_embedding)`.

        Returns:
            torch.Tensor: Predicted torsion angles as (cos, sin) pairs.
                Shape: `(..., number_residues, number_torsion_angles, 2)`.
        """
        output = (self.current_single_representation_embedder(self.relu(single_representation)) +
                  self.initial_single_representation_embedder(self.relu(initial_single_representation)))

        for layer in self.angle_resnet_layers:
            output = layer(output)

        residue_angles_unstacked = self.torsion_angles_output_embedder(self.relu(output))

        residue_angles_chunks = torch.split(tensor=residue_angles_unstacked, split_size_or_sections=2, dim=-1)
        residue_angles = torch.stack(residue_angles_chunks, dim=-2)

        return residue_angles


class StructureModule(nn.Module):
    """
    The final component of AlphaFold II that predicts 3D coordinates from single and pair representations.
    """

    def __init__(self, single_representation_embedding: int, pair_representation_embedding: int,
                 number_layers: int, angle_representation_embedding: int,
                 number_query_points: int, number_value_points: int,
                 number_heads: int, head_embedding_dimension: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the StructureModule.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            pair_representation_embedding (int): Feature dimension of the pair representation.
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
            number_layers (int): Number of iterative updates.
            angle_representation_embedding (int): Dimension for angle ResNet.
            number_query_points (int): Number of geometric query points for IPA.
            number_value_points (int): Number of geometric value points for IPA.
            number_heads (int): Number of attention heads for IPA.
            head_embedding_dimension (int): Hidden dimension per head for IPA.
        """
        super().__init__()

        self.single_representation_embedding = single_representation_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.angle_representation_embedding = angle_representation_embedding
        self.number_layers = number_layers
        self.device = device
        self.dtype = dtype
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_heads = number_heads
        self.head_embedding_dimension = head_embedding_dimension

        self.single_representation_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.single_representation_embedding,
            device=self.device, dtype=self.dtype)
        self.pair_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.pair_representation_embedding,
                                                                 device=self.device, dtype=self.dtype)
        self.initial_single_representation_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                                                out_features=self.single_representation_embedding,
                                                                device=self.device, dtype=self.dtype)

        self.invariant_point_attention_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.single_representation_embedding, device=self.device, dtype=self.dtype)

        # Initializing IPA
        self.invariant_point_attention = InvariantPointAttention(
            single_representation_embedding=self.single_representation_embedding,
            pair_representation_embedding=self.pair_representation_embedding,
            number_query_points=self.number_query_points,
            number_value_points=self.number_value_points,
            number_heads=self.number_heads,
            head_embedding_dimension=self.head_embedding_dimension,
            device=self.device,
            dtype=self.dtype)

        self.structure_module_transition = StructureModuleTransition(
            single_representation_embedding=self.single_representation_embedding,
            device=self.device, dtype=self.dtype)

        self.backbone_update = BackboneUpdate(single_representation_embedding=self.single_representation_embedding,
                                              device=self.device, dtype=self.dtype)

        self.angle_resnet = AngleResNet(single_representation_embedding=self.single_representation_embedding,
                                        angle_representation_embedding=self.angle_representation_embedding,
                                        device=self.device, dtype=self.dtype)

    def process_outputs(self, transformation_matrix: torch.Tensor, residue_angles: torch.Tensor,
                        sequence_amino_acid_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Post-processes predicted frames and angles to obtain final 3D coordinates.

        Args:
            transformation_matrix (torch.Tensor): Global backbone transformations.
                Shape: `(..., number_residues, 4, 4)`.
            residue_angles (torch.Tensor): Predicted torsion angles.
                Shape: `(..., number_residues, number_torsion_angles, 2)`.
            sequence_amino_acid_labels (torch.Tensor): Amino acid type indices.
                Shape: `(..., number_residues)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - final_positions: Coordinates for all 37 atoms. Shape `(..., number_residues, 37, 3)`.
                - position_mask: Mask for present atoms. Shape `(..., number_residues, 37)`.
                - pseudo_beta_positions: Predicted C-beta (or C-alpha for Glycine) positions.
                 Shape `(..., number_residues, 3)`.
        """

        # Scale translations before coordinate calculation (nanometers to angstroms)
        transformation_matrix_clone = transformation_matrix.clone()
        transformation_matrix_clone[..., :3, -1] = 10 * transformation_matrix_clone[..., :3, -1]

        # Final positions : (..., number_residues, 37, 3)
        # Position Masks : (..., number_residues, 37)
        final_positions, position_mask = compute_all_atom_coordinates(
            transformation_matrix=transformation_matrix_clone,
            residue_angles=residue_angles,
            sequence_amino_acid_labels=sequence_amino_acid_labels)

        # We use carbon beta atoms for all amino acids except for glycine
        carbon_beta_index = atom_types.index("CB")
        carbon_alpha_index = atom_types.index("CA")
        glycine_index = canonical_amino_acid_residues.index("G")

        # Pseudo Beta Positions : (..., number_residues, 3)
        pseudo_beta_positions = final_positions[..., carbon_beta_index, :]
        alpha_positions = final_positions[..., carbon_alpha_index, :]
        pseudo_beta_positions[sequence_amino_acid_labels == glycine_index] = alpha_positions[
            sequence_amino_acid_labels == glycine_index]

        return final_positions, position_mask, pseudo_beta_positions

    def forward(self, single_representation: torch.Tensor,
                pair_representation: torch.Tensor,
                sequence_amino_acid_labels: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the iterative Structure Module pipeline to predict 3D protein structure.

        This method performs several iterations of geometric updates. In each iteration,
        it applies Invariant Point Attention (IPA) and a transition layer to the single
        representation, updates the backbone frames, and predicts torsion angles.
        The final backbone frames and angles are used to compute the all-atom coordinates.

        Args:
            single_representation (torch.Tensor): Residue-level features.
                Shape: `(..., number_residues, single_representation_dimension)`.
            pair_representation (torch.Tensor): Pairwise features.
                Shape: `(..., number_residues, number_residues, pair_representation_dimension)`.
            sequence_amino_acid_labels (torch.Tensor): Amino acid type indices.
                Shape: `(..., number_residues)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - angles: Predicted torsion angles as (cos, sin) pairs for all iterations.
                  Shape: `(..., number_layers, number_residues, 7, 2)`.
                - frames: Global backbone transformation matrices for all iterations.
                  Shape: `(..., number_layers, number_residues, 4, 4)`.
                - final_positions: Final 3D coordinates for all 37 atom types.
                  Shape: `(..., number_residues, 37, 3)`.
                - position_mask: Binary mask indicating the presence of each atom in the final positions.
                  Shape: `(..., number_residues, 37)`.
                - pseudo_beta_positions: Predicted positions of C-beta atoms (or C-alpha for Glycine).
                  Shape: `(..., number_residues, 3)`.
        """
        number_residues = pair_representation.shape[-2]
        batch_dimension = single_representation.shape[:-2]
        outputs = {'angles': [], 'frames': []}
        device = single_representation.device
        dtype = single_representation.dtype

        # initial single representation never changed after this, just passed around
        initial_single_representation = self.single_representation_layer_normalizer(single_representation)

        # These pair_representation and single_representations are modified in the for loop
        pair_representation = self.pair_representation_layer_normalizer(pair_representation)
        single_representation = self.initial_single_representation_embedder(single_representation)

        # Initial transformation matrix as an identity matrix.
        transformation_matrix = (torch.eye(4, device=device, dtype=dtype).
                                 broadcast_to(batch_dimension + (number_residues, 4, 4)))

        # TODO NOTE : IT IS IN THIS BLOCK WHERE WE WILL BE INSERTING LOSSES FOR BACKPROPAGATION
        for iteration in range(self.number_layers):
            # IPA and it's normalizer
            # TODO It would have been better to normalise the input of the invariant point attention before adding it
            #  it is unusually large compared to the incoming single representation
            single_representation += self.invariant_point_attention(single_representation=single_representation,
                                                                    pair_representation=pair_representation,
                                                                    transformation_matrix=transformation_matrix)
            single_representation = self.invariant_point_attention_layer_normalizer(single_representation)

            # Transition and it's normalizer(included in the transition layer)
            single_representation = self.structure_module_transition(single_representation)

            # Update of the transformation matrix by right multiplication of current predicted transformation
            # Here the current one being << self.backbone_update(single_representation) >>
            transformation_matrix = torch.matmul(input=transformation_matrix,
                                                 other=self.backbone_update(single_representation))

            # Prediction of residue angles (..., number_residues, 7, 2)
            residue_angles = self.angle_resnet(single_representation=single_representation,
                                               initial_single_representation=initial_single_representation)

            outputs['angles'].append(residue_angles)
            outputs['frames'].append(transformation_matrix)

        angles = torch.stack(outputs['angles'], dim=-4)
        frames = torch.stack(outputs['frames'], dim=-4)

        final_positions, position_mask, pseudo_beta_positions = self.process_outputs(
            transformation_matrix=transformation_matrix,
            residue_angles=residue_angles,
            sequence_amino_acid_labels=sequence_amino_acid_labels)

        return angles, frames, final_positions, position_mask, pseudo_beta_positions
