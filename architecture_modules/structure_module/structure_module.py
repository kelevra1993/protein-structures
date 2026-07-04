import torch
import time
from torch import nn
from typing import Tuple, Optional

from architecture_modules.lddt_module.lddt_module import LddtModule
from architecture_modules.structure_module.invariant_point_attention_module import InvariantPointAttention
from utilities.geometry_utilities import compute_all_atom_coordinates, assemble_4x4_transform_matrix, \
    turn_quaternion_to_3x3_matrix

# Here we will have to design this explicitly
from utilities.constants import atom_types, canonical_amino_acid_residues, index_to_xxx, chi_angles_mask
from utilities.loss_utilities import compute_fape_loss, compute_torsion_angle_loss, \
    rename_symmetric_ground_truth_metrics, compute_local_distance_difference_test, compute_plddt_loss
from utilities.tensor_utilities import print_tensor_shape, print_tensor_list


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
                Shape: `(..., number_residues, single_representation_dimension)`.

        Returns:
            torch.Tensor: Updated single representation.
                Shape: `(..., number_residues, single_representation_dimension)`.
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
                Shape: `(..., number_residues, angle_representation_embedding)`.

        Returns:
            torch.Tensor: Updated angle representation.
                Shape: `(..., number_residues, angle_representation_embedding)`.
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
                Shape: `(..., number_residues, single_representation_dimension)`.
            initial_single_representation (torch.Tensor): Initial single representation features.
                Shape: `(..., number_residues, single_representation_dimension)`.

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
                 number_iterations: int, angle_representation_embedding: int,
                 number_query_points: int, number_value_points: int,
                 number_heads: int, head_embedding_dimension: int,
                 number_torsion_angles: int, device: torch.device, dtype: torch.dtype,
                 unclamp_fape_ratio: float = 0.1, enable_side_chain_fape_loss: bool = True, 
                 enable_lddt_loss: bool = True, clamp_fape_threshold: float = 10.0):
        """
        Initializes the StructureModule.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            pair_representation_embedding (int): Feature dimension of the pair representation.
            number_iterations (int): Number of iterative updates. Equivalent to re-usage of layer
            angle_representation_embedding (int): Dimension for angle ResNet.
            number_query_points (int): Number of geometric query points for IPA.
            number_value_points (int): Number of geometric value points for IPA.
            number_heads (int): Number of attention heads for IPA.
            head_embedding_dimension (int): Hidden dimension per head for IPA.
            number_torsion_angles (int): Number of torsion angles to predict per residue.
            device (torch.device): Device for tensor allocation.
            dtype (torch.dtype): Data type for tensors.
            unclamp_fape_ratio (float): The probability to unclamp the FAPE loss.
            enable_side_chain_fape_loss (bool): Whether to compute the all-atom/side-chain FAPE loss.
            enable_lddt_loss (bool): Whether to compute the pLDDT loss.
            clamp_fape_threshold (float): The distance threshold (in Angstroms) for FAPE clamping.
        """
        super().__init__()

        self.single_representation_embedding = single_representation_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.angle_representation_embedding = angle_representation_embedding
        self.number_iterations = number_iterations
        self.device = device
        self.dtype = dtype
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_heads = number_heads
        self.head_embedding_dimension = head_embedding_dimension
        self.number_torsion_angles = number_torsion_angles
        self.unclamp_fape_ratio = unclamp_fape_ratio
        self.enable_side_chain_fape_loss = enable_side_chain_fape_loss
        self.enable_lddt_loss = enable_lddt_loss
        self.clamp_fape_threshold = clamp_fape_threshold
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
                                        number_torsion_angles=self.number_torsion_angles,
                                        device=self.device, dtype=self.dtype)

        # Initializing Local Distance Difference Test Prediction Module
        self.lddt_module = LddtModule(single_representation_embedding=self.single_representation_embedding,
                                      intermediate_embedding=int(self.single_representation_embedding / 4),
                                      device=self.device, dtype=self.dtype)

    @staticmethod
    def process_outputs(
            transformation_matrix: torch.Tensor,
            residue_angles: torch.Tensor,
            sequence_amino_acid_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Post-processes predicted frames and angles to obtain final 3D coordinates and global transformation matrices.

        Args:
            transformation_matrix (torch.Tensor): Global backbone transformations.
                Shape: `(..., number_residues, 4, 4)`.
            residue_angles (torch.Tensor): Predicted torsion angles.
                Shape: `(..., number_residues, number_torsion_angles, 2)`.
            sequence_amino_acid_labels (torch.Tensor): Amino acid type indices.
                Shape: `(..., number_residues)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - final_positions: Coordinates for all 37 atoms. Shape `(..., number_residues, 37, 3)`.
                - position_mask: Mask for present atoms. Shape `(..., number_residues, 37)`.
                - pseudo_beta_positions: Predicted C-beta (or C-alpha for Glycine) positions.
                  Shape `(..., number_residues, 3)`.
                - global_transformation_matrices: Transformation matrices for all 8 rigid groups.
                  Shape `(..., number_residues, 8, 4, 4)`.
        """

        # Final positions : (..., number_residues, 37, 3)
        # Position Masks : (..., number_residues, 37)
        # Global Transformation Matrices : (..., number_residues, 37, 8, 4, 4)
        final_positions, position_mask, global_transformation_matrices = compute_all_atom_coordinates(
            transformation_matrix=transformation_matrix,
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

        return final_positions, position_mask, pseudo_beta_positions, global_transformation_matrices

    def forward(self,
                single_representation: torch.Tensor,
                pair_representation: torch.Tensor,
                sequence_amino_acid_labels: torch.Tensor,
                ground_truth_transformation_matrix: torch.Tensor,
                alternative_ground_truth_transformation_matrix: torch.Tensor,
                ground_truth_angles: torch.Tensor,
                alternative_ground_truth_angles: torch.Tensor,
                ground_truth_positions: torch.Tensor,
                alternative_ground_truth_positions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the iterative Structure Module pipeline to predict 3D protein structure.

        This method performs several iterations of geometric updates. In each iteration,
        it applies Invariant Point Attention (IPA) and a transition layer to the single
        representation, updates the backbone frames, and predicts torsion angles.
        The final backbone frames and angles are used to compute the all-atom coordinates
        and structural losses.

        Args:
            single_representation (torch.Tensor): Residue-level features.
                Shape: `(..., number_residues, single_representation_dimension)`.
            pair_representation (torch.Tensor): Pairwise features.
                Shape: `(..., number_residues, number_residues, pair_representation_dimension)`.
            sequence_amino_acid_labels (torch.Tensor): Amino acid type indices.
                Shape: `(..., number_residues)`.
            ground_truth_transformation_matrix (torch.Tensor): Ground truth global backbone
                transformation matrices. Shape: `(..., number_residues, 8, 4, 4)`.
            alternative_ground_truth_transformation_matrix (torch.Tensor): Alternative ground truth
                global backbone transformation matrices (e.g., for symmetric cases).
                Shape: `(..., number_residues, 8, 4, 4)`.
            ground_truth_angles (torch.Tensor): Ground truth torsion angles.
                Shape: `(..., number_residues, 7, 2)`.
            alternative_ground_truth_angles (torch.Tensor): Alternative ground truth torsion angles.
                Shape: `(..., number_residues, 7, 2)`.
            ground_truth_positions (torch.Tensor): Ground truth 3D coordinates for all atoms.
                Shape: `(..., number_residues, 37, 3)`.
            alternative_ground_truth_positions (torch.Tensor): Alternative ground truth 3D coordinates.
                Shape: `(..., number_residues, 37, 3)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                - overall_fape_loss: Computed Frame Aligned Point Error (FAPE) loss across all atoms.
                  Shape: `()`.
                - auxillary_loss: Averaged auxiliary loss across all iterations (FAPE and torsion angle).
                  Shape: `()`.
                - predicted_lddt_loss: Computed predicted Local Distance Difference Test (pLDDT) loss.
                  Shape: `()`.
                - true_lddt: The actual calculated lDDT score per residue.
                  Shape: `(..., number_residues)`.
                - unclamped_fape: The true average distance error in Angstroms (unclamped FAPE).
                  Shape: `()`.
        """
        number_residues = pair_representation.shape[-2]
        batch_dimension = single_representation.shape[:-2]
        outputs = {'angles': [], 'frames': []}
        device = single_representation.device
        dtype = single_representation.dtype

        # Setup the batch size
        if list(batch_dimension):
            batch_size = batch_dimension[0]
        else:
            batch_size = 1
            single_representation = single_representation.unsqueeze(0)
            pair_representation = pair_representation.unsqueeze(0)
            sequence_amino_acid_labels = sequence_amino_acid_labels.unsqueeze(0)
            ground_truth_transformation_matrix = ground_truth_transformation_matrix.unsqueeze(0)
            alternative_ground_truth_transformation_matrix = alternative_ground_truth_transformation_matrix.unsqueeze(0)
            ground_truth_angles = ground_truth_angles.unsqueeze(0)
            alternative_ground_truth_angles = alternative_ground_truth_angles.unsqueeze(0)
            ground_truth_positions = ground_truth_positions.unsqueeze(0)
            alternative_ground_truth_positions = alternative_ground_truth_positions.unsqueeze(0)

        # Get the Backbone Transformation Matrix from (..., number_residues, 8, 4, 4) -> (..., number_residues, 4, 4)`
        ground_truth_backbone_transformation_matrix = ground_truth_transformation_matrix[..., 0, :, :]

        # Get ground truth carbon alpha positions
        # Note we could also have got it from the translation matrix of the backbone transformation
        carbon_alpha_index = atom_types.index("CA")
        ground_truth_carbon_alpha_positions = ground_truth_positions[..., carbon_alpha_index, :]

        # initial single representation never changed after this, just passed around
        initial_single_representation = self.single_representation_layer_normalizer(single_representation)

        # These pair_representation and single_representations are modified in the for loop
        pair_representation = self.pair_representation_layer_normalizer(pair_representation)
        single_representation = self.initial_single_representation_embedder(single_representation)

        # Initial backbone transformation matrix as an identity matrix.
        transformation_matrix = (torch.eye(4, device=device, dtype=dtype).
                                 broadcast_to(batch_dimension + (number_residues, 4, 4)))

        # Determine backbone clamping for this batch based on AlphaFold II paper:
        # "In 90% of training mini-batches the FAPE backbone loss is clamped by emax = 10 A, 
        # in the remaining 10% it is not clamped, emax = +inf. For side-chains it is always clamped by emax = 10 A."
        clamp_current_backbone = torch.rand(1).item() >= self.unclamp_fape_ratio
        backbone_distance_clamp = self.clamp_fape_threshold if clamp_current_backbone else 1e10
        sidechain_distance_clamp = self.clamp_fape_threshold

        # Losses
        auxillary_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Precompute the angle mask based on the amino acid sequence
        chi_mask_tensor = torch.tensor(chi_angles_mask, device=device, dtype=dtype)
        backbone_mask = torch.ones((21, 3), device=device, dtype=dtype)
        full_angle_mask = torch.cat([backbone_mask, chi_mask_tensor], dim=-1)
        batch_angle_mask = full_angle_mask[sequence_amino_acid_labels]

        # Equivalent to re-usage of layer
        for iteration in range(self.number_iterations):
            # IPA and it's normalizer
            # TODO It would have been better to normalise the output of the invariant point attention before adding it
            #  it is unusually large compared to the incoming single representation.
            #  Very important when we will start training
            single_representation = single_representation + self.invariant_point_attention(
                single_representation=single_representation,
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
            # Here they have not yet been normalized
            residue_angles = self.angle_resnet(single_representation=single_representation,
                                               initial_single_representation=initial_single_representation)

            # We Compute Fape Loss With Carbon Alpha Positions and We Compute Torsion Angle Loss
            # First get backbone frame rotation and translation after transformation update
            # Note : carbon alpha positions correspond to the translation_matrix since we are dealing with the backbone
            rotation_matrix = transformation_matrix[..., :3, :3]
            translation_matrix = transformation_matrix[..., :3, -1]

            # Note we are only using the backbone transformation matrices
            # Note we are only using carbon alpha global positions as inputs
            iteration_fape_loss, _ = compute_fape_loss(
                predicted_transformation_matrix=transformation_matrix,
                predicted_positions=translation_matrix,
                ground_truth_transformation_matrix=ground_truth_backbone_transformation_matrix,
                ground_truth_positions=ground_truth_carbon_alpha_positions,
                distance_clamp=backbone_distance_clamp)

            #  DON'T FORGET TO RE-CHECK THE ANGLE NORM LOSS SCALER AND ADD IT TO TENSORBOARD
            # Be careful, ground truth angle should already be normalised in the form (cos(phi), sin(phi))
            iteration_torsion_angle_loss = compute_torsion_angle_loss(
                predicted_unnormalised_angles=residue_angles,
                ground_truth_angles=ground_truth_angles,
                alternative_ground_truth_angles=alternative_ground_truth_angles,
                mask=batch_angle_mask,
                angle_norm_loss_scaler=0.02)

            # Sum up the losses for this iteration and add them to auxillary loss
            # We average them over the batch dimensions before adding them to the auxillary loss
            iteration_auxillary_loss = torch.mean(iteration_fape_loss + iteration_torsion_angle_loss)
            auxillary_loss = auxillary_loss + iteration_auxillary_loss

            # No rotation gradients between iterations to stabilize training except for the last iteration
            # Using .detach() on rotation_matrix
            if iteration < self.number_iterations - 1:
                transformation_matrix = assemble_4x4_transform_matrix(rotation_matrix=rotation_matrix.detach(),
                                                                      translation_vector=translation_matrix)

            # We normalize angles to [cos(angle),sin(angle)] for model outputs
            outputs['angles'].append(torch.nn.functional.normalize(residue_angles, dim=-1))
            outputs['frames'].append(transformation_matrix)

        # Average Out The Auxillary Loss
        auxillary_loss = auxillary_loss / self.number_iterations

        angles = torch.stack(outputs['angles'], dim=-4)
        frames = torch.stack(outputs['frames'], dim=-4)

        # We only use the last residue angles (here they are not yet normalised)
        final_positions, position_mask, pseudo_beta_positions, global_transformation_matrices = self.process_outputs(
            transformation_matrix=transformation_matrix,
            residue_angles=residue_angles,
            sequence_amino_acid_labels=sequence_amino_acid_labels)

        # Implementation of the renaming of the symmetric ground truth atoms
        # Since we are dealing with only heavy atoms and some residue present areas of 180 symetry rotation
        # the network can predict coordinates that are legit and therefore it should not be penalized,
        # therefore the ground truth has to change accordingly, but it has to be done at every prediction cycle

        for batch_index in range(batch_size):
            # Modify Ground Truth Positions And Frames Accordingly, by iterating through batches
            (ground_truth_positions[batch_index],
             ground_truth_transformation_matrix[batch_index]) = rename_symmetric_ground_truth_metrics(
                predicted_positions=final_positions[batch_index],
                ground_truth_transformation_matrix=ground_truth_transformation_matrix[batch_index],
                ground_truth_positions=ground_truth_positions[batch_index],
                alternative_ground_truth_transformation_matrix=alternative_ground_truth_transformation_matrix[
                    batch_index],
                alternative_ground_truth_positions=alternative_ground_truth_positions[batch_index],
                sequence_amino_acid_labels=sequence_amino_acid_labels[batch_index])

        # Note Here we use all frames in the transformation matrices
        # Note We also use all positions
        overall_fape_loss = torch.tensor(0.0, device=device, dtype=dtype)
        unclamped_fape_metric = torch.tensor(0.0, device=device, dtype=dtype)
        fape_loss_counter = torch.tensor(1e-8, device=device, dtype=dtype)

        # The computation of the side chain fape loss takes alot of time
        # We might initially want to ingore it and only launch it at later stages of refining structure.
        if self.enable_side_chain_fape_loss:
            for frame_index in range(8):
    
                # Get the Current Frame Matrix from (..., number_residues, 8, 4, 4) -> (..., number_residues, 4, 4)`
                current_predicted_frame = global_transformation_matrices[..., frame_index, :, :]
                current_ground_truth_frame = ground_truth_transformation_matrix[..., frame_index, :, :]
    
                # Go through all atom types
                for atom_index in range(37):
    
                    # Implementation of the mask of size (batch, number_residues)
                    current_position_masks = position_mask[..., atom_index]
    
                    # If all the atom is not present for any residue in the batch, do not compute fape loss
                    if not current_position_masks.any():
                        continue
    
                    # Get ground truth current positions
                    current_predicted_positions = final_positions[..., atom_index, :]
                    current_ground_truth_positions = ground_truth_positions[..., atom_index, :]
    
                    # In AlphaFold 2, backbone FAPE uses backbone frames (frame 0) and backbone atoms (N, CA, C, CB, O)
                    # Atom types list indices: 0:N, 1:CA, 2:C, 3:CB, 4:O
                    is_backbone_fape = (frame_index == 0) and (atom_index in [0, 1, 2, 3, 4])
                    side_chain_distance_clamp = backbone_distance_clamp if is_backbone_fape else sidechain_distance_clamp
    
                    # Call the loss with a mask
                    frame_atom_fape_loss, frame_atom_unclamped_fape = compute_fape_loss(
                        predicted_transformation_matrix=current_predicted_frame,
                        predicted_positions=current_predicted_positions,
                        mask=current_position_masks,
                        ground_truth_transformation_matrix=current_ground_truth_frame,
                        ground_truth_positions=current_ground_truth_positions,
                        distance_clamp=side_chain_distance_clamp)
    
                    overall_fape_loss = overall_fape_loss + torch.mean(frame_atom_fape_loss)
                    unclamped_fape_metric = unclamped_fape_metric + torch.mean(frame_atom_unclamped_fape)
                    fape_loss_counter = fape_loss_counter + 1.0
    
            # Average out the overall fape loss by the number of present positions
            overall_fape_loss = overall_fape_loss / fape_loss_counter
            unclamped_fape_metric = unclamped_fape_metric / fape_loss_counter

        # Implementation of Prediction Per Residue LDDT C-Alpha Loss
        local_difference_distance_test = compute_local_distance_difference_test(
            prediction_positions=final_positions,
            ground_truth_positions=ground_truth_positions)

        # Predict LDDT Logits and Probabilities as well as
        if self.enable_lddt_loss:
            lddt_logits, lddt_probabilities, predicted_lddt_per_residue = self.lddt_module(
                single_representation=single_representation)

            # Compute LDDT Loss
            predicted_lddt_loss = compute_plddt_loss(ground_truth_lddt=local_difference_distance_test,
                                                     predicted_lddt_logits=lddt_logits,
                                                     lddt_bins=self.lddt_module.lddt_bins)
        else:
            predicted_lddt_loss = torch.tensor(0.0, device=device, dtype=dtype)

        return (angles, frames, final_positions, position_mask, pseudo_beta_positions, overall_fape_loss,
                auxillary_loss, predicted_lddt_loss, local_difference_distance_test, unclamped_fape_metric)
