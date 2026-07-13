import torch
from torch import nn
from typing import Dict

from embedders.input_embedder import InputEmbedder
from embedders.recycling_embedder import RecyclingEmbedder
from embedders.extra_msa_embedder import ExtraMsaStack, ExtraMsaEmbedder
from architecture_modules.evoformer_module.evoformer import EvoformerStack
from architecture_modules.structure_module.structure_module import StructureModule
from architecture_modules.distogram_module.distogram_module import DistogramModule
from utilities.loss_utilities import compute_distogram_loss


class Model(nn.Module):

    def __init__(self, configuration: dict, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Alphafold model using a configuration dictionary.

        Args:
            configuration (dict): A dictionary parsed from a YAML configuration file.
            device (torch.device, optional): Device on which the model should run.
            dtype (torch.dtype, optional): Data type of the model.
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        global_configuration = configuration.get('GlobalConfiguration', {})
        input_embedder_configuration = configuration.get('InputEmbedder', {})
        extra_msa_stack_configuration = configuration.get('ExtraMsaStack', {})
        evoformer_stack_configuration = configuration.get('EvoformerStack', {})
        structure_module_configuration = configuration.get('StructureModule', {})

        # Kept for usage in the forward pass
        self.msa_embedding = global_configuration.get('msa_embedding')
        self.pair_representation_embedding = global_configuration.get('pair_representation_embedding')
        self.extra_msa_embedding = global_configuration.get('extra_msa_embedding')
        self.single_representation_embedding = global_configuration.get('single_representation_embedding')
        self.enable_distogram_loss = global_configuration.get('enable_distogram_loss')

        self.input_embedder = InputEmbedder(
            input_sequence_feature_dimension=global_configuration.get('input_sequence_feature_dimension'),
            input_msa_feature_dimension=global_configuration.get('input_msa_feature_dimension'),
            msa_embedding=global_configuration.get('msa_embedding'),
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            number_neighbouring_amino_acids=input_embedder_configuration.get('number_neighbouring_amino_acids'),
            device=self.device,
            dtype=self.dtype)

        self.extra_msa_embedder = ExtraMsaEmbedder(
            input_extra_msa_feature_dimension=global_configuration.get('input_extra_msa_feature_dimension'),
            extra_msa_embedding=global_configuration.get('extra_msa_embedding'),
            device=self.device,
            dtype=self.dtype)

        self.recycling_embedder = RecyclingEmbedder(
            msa_embedding=global_configuration.get('msa_embedding'),
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            device=self.device,
            dtype=self.dtype)

        self.extra_msa_stack = ExtraMsaStack(
            extra_msa_embedding=global_configuration.get('extra_msa_embedding'),
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            number_blocks=extra_msa_stack_configuration.get('number_blocks'),
            msa_number_heads=extra_msa_stack_configuration.get('msa_number_heads'),
            msa_head_embedding_dimension=extra_msa_stack_configuration.get('msa_head_embedding_dimension'),
            msa_global_number_heads=extra_msa_stack_configuration.get('msa_global_number_heads'),
            msa_global_head_embedding_dimension=extra_msa_stack_configuration.get(
                'msa_global_head_embedding_dimension'),
            pair_number_heads=extra_msa_stack_configuration.get('pair_number_heads'),
            pair_head_embedding_dimension=extra_msa_stack_configuration.get('pair_head_embedding_dimension'),
            intermediate_embedding=extra_msa_stack_configuration.get('intermediate_embedding'),
            msa_transition_channel_scaler=extra_msa_stack_configuration.get('msa_transition_channel_scaler'),
            pair_stack_channel_scaler=extra_msa_stack_configuration.get('pair_stack_channel_scaler'),
            triangle_multiplication_embedding=extra_msa_stack_configuration.get('triangle_multiplication_embedding'),
            device=self.device,
            dtype=self.dtype)

        self.evoformer = EvoformerStack(
            msa_embedding=global_configuration.get('msa_embedding'),
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            number_blocks=evoformer_stack_configuration.get('number_blocks'),
            msa_number_heads=evoformer_stack_configuration.get('msa_number_heads'),
            msa_head_embedding_dimension=evoformer_stack_configuration.get('msa_head_embedding_dimension'),
            pair_number_heads=evoformer_stack_configuration.get('pair_number_heads'),
            pair_head_embedding_dimension=evoformer_stack_configuration.get('pair_head_embedding_dimension'),
            msa_transition_channel_scaler=evoformer_stack_configuration.get('msa_transition_channel_scaler'),
            pair_stack_channel_scaler=evoformer_stack_configuration.get('pair_stack_channel_scaler'),
            intermediate_embedding=evoformer_stack_configuration.get('intermediate_embedding'),
            triangle_multiplication_embedding=evoformer_stack_configuration.get('triangle_multiplication_embedding'),
            single_representation_embedding=global_configuration.get('single_representation_embedding'),
            device=self.device,
            dtype=self.dtype)

        self.structure_module = StructureModule(
            single_representation_embedding=global_configuration.get('single_representation_embedding'),
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            number_iterations=structure_module_configuration.get('number_structure_module_iterations'),
            angle_representation_embedding=structure_module_configuration.get('angle_representation_embedding'),
            number_query_points=structure_module_configuration.get('number_query_points'),
            number_value_points=structure_module_configuration.get('number_value_points'),
            number_heads=structure_module_configuration.get('number_heads'),
            head_embedding_dimension=structure_module_configuration.get('head_embedding_dimension'),
            number_torsion_angles=structure_module_configuration.get('number_torsion_angles'),
            unclamp_fape_ratio=global_configuration.get('unclamp_fape_ratio'),
            enable_side_chain_fape_loss=global_configuration.get('enable_side_chain_fape_loss'),
            enable_lddt_loss=global_configuration.get('enable_lddt_loss'),
            clamp_fape_threshold=global_configuration.get('clamp_fape_threshold'),
            use_peptide_linker_module=structure_module_configuration.get('use_peptide_linker_module'),
            peptide_linker_representation_embedding=structure_module_configuration.get('peptide_linker_representation_embedding'),
            device=self.device,
            dtype=self.dtype)

        self.distogram_module = DistogramModule(
            pair_representation_embedding=global_configuration.get('pair_representation_embedding'),
            device=self.device,
            dtype=self.dtype)

    def forward(self, batch_input_dictionary: Dict[str, torch.Tensor]):
        """
        Forward pass for the Alphafold model ran for multiple cycles.
        The msa and extra msa features change in each cycle
        Args:
        batch_input_dictionary (dict): A dictionary containing the following features:
         * input_msa_feature:
         Tensor of shape (*, number_clusters, number_residues, msa_feature_dimension, number_cycles).
         * input_extra_msa_feature:
         Tensor of shape (*, number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_cycles).
         * input_sequence_feature:
         Tensor of shape (*, number_residues, input_sequence_feature_dimension).
         One-hot encoding of the target sequence.
         * input_residue_index_feature:
         Tensor of shape (*, number_residues).
         The absolute index of each residue.
         * ground_truth_frames: Shape (*, number_residues, 8, 4, 4)
         * alternative_ground_truth_frames: Shape (*, number_residues, 8, 4, 4)
         * ground_truth_angles: Shape (*, number_residues, 7, 2)
         * alternative_ground_truth_angles: Shape (*, number_residues, 7, 2)
         * ground_truth_global_positions: Shape (*, number_residues, 37, 3)
         * alternative_ground_truth_global_positions: Shape (*, number_residues, 37, 3)
         * distogram_labels: Shape (*, number_residues, number_residues)

        Returns:
        dict: A dictionary with the following entries:
            * final_positions: Heavy-atom positions in Angstrom of shape (*, number_residues, 37, 3, number_cycles).
            * position_mask: Boolean tensor of shape (*, number_residues, 37, number_cycles), masking atoms that
                aren't present in the amino acids.
            * angles: Torsion angles of shape (*, num_layers, number_residues, number_torsion_angles, 2, number_cycles)
             for every iteration of the Structure Module in every cycle.
            * frames: Backbone frames of shape (*, num_layers, number_residues, 4, 4, number_cycles)
             for every iteration of the Structure Module in every cycle.
            * pseudo_beta_positions: Pseudo C-beta positions of shape (*, number_residues, 3, number_cycles).
            * overall_fape_loss: FAPE loss.
            * auxillary_loss: Auxiliary loss.
            * predicted_lddt_loss: Predicted LDDT loss.
            * true_lddt: Actual calculated lDDT score per residue.
            * unclamped_fape: The true average distance error in Angstroms (unclamped FAPE).
            * distogram_logits: Distance bin logits of shape (*, number_residues, number_residues, 64).
        """

        number_cycles = batch_input_dictionary['input_msa_feature'].shape[-1]
        number_clusters, number_residues = batch_input_dictionary['input_msa_feature'].shape[-4:-2]
        batch_shape = batch_input_dictionary['input_msa_feature'].shape[:-4]

        # Model outputs emanating from structure module
        model_outputs = {key: [] for key in
                         ["angles", "frames", "final_positions", "position_mask", "pseudo_beta_positions",
                          "overall_fape_loss", "auxillary_loss", "predicted_lddt_loss", "true_lddt", "unclamped_fape",
                          "peptide_linker_scalers"]}

        # Initialisation of first tensors
        msa_shape = (batch_shape + (number_clusters, number_residues, self.msa_embedding))
        previous_msa_representation_tensor = torch.zeros(msa_shape, dtype=self.dtype, device=self.device)

        pair_shape = (batch_shape + (number_residues, number_residues, self.pair_representation_embedding))
        previous_pair_representation_tensor = torch.zeros(pair_shape, dtype=self.dtype, device=self.device)

        position_shape = (batch_shape + (number_residues, 3))
        previous_pseudo_carbon_beta_positions = torch.zeros(position_shape, dtype=self.dtype, device=self.device)

        for cycle in range(number_cycles):
            # Extract Current Input Features For This Cycle
            current_cycle_input_batch = {}
            for key, value in batch_input_dictionary.items():
                if key in ["input_msa_feature", "input_extra_msa_feature"]:
                    current_cycle_input_batch[key] = value[..., cycle]
                else:
                    current_cycle_input_batch[key] = value

            # Get embeddings for msa and pair representation
            msa_representation_tensor, pair_representation_tensor = self.input_embedder(
                input_sequence_feature=current_cycle_input_batch['input_sequence_feature'],
                input_msa_feature=current_cycle_input_batch['input_msa_feature'],
                input_residue_index_feature=current_cycle_input_batch['input_residue_index_feature'])

            # Run the model through the recycling embedder
            recycled_msa_representation, recycled_pair_representation = self.recycling_embedder(
                previous_msa_representation=previous_msa_representation_tensor,
                previous_pair_representation=previous_pair_representation_tensor,
                previous_pseudo_carbon_beta_positions=previous_pseudo_carbon_beta_positions)

            # Only the very first sequence of the msa is updated with the recycled msa
            # The recycling embedder just actually normalizes the msa_representation_input using a layer normalizer

            # Non-inplace update for msa_representation_tensor
            first_msa_sequence = msa_representation_tensor[..., 0, :, :] + recycled_msa_representation
            msa_representation_tensor = torch.cat([first_msa_sequence.unsqueeze(-3),
                                                   msa_representation_tensor[..., 1:, :, :]], dim=-3)

            pair_representation_tensor = pair_representation_tensor + recycled_pair_representation

            extra_msa_representation = self.extra_msa_embedder(
                input_extra_msa_feature=current_cycle_input_batch['input_extra_msa_feature'])

            # The pair representation is updated by the extra msa embedding before being fed to the evoformer stack
            pair_representation_tensor = self.extra_msa_stack(
                extra_msa_representation=extra_msa_representation,
                pair_representation=pair_representation_tensor)

            # Pass through the evoformer block
            msa_representation_tensor, pair_representation_tensor, single_representation_tensor = self.evoformer(
                msa_representation=msa_representation_tensor,
                pair_representation=pair_representation_tensor)

            # Sequence amino acid labels used for :
            # - selecting backbones
            # - computing atom coordinates
            # - computing masks
            sequence_amino_acid_labels = current_cycle_input_batch["sequence_labels"]

            (angles, frames, final_positions, position_mask, pseudo_beta_positions,
             overall_fape_loss, auxillary_loss, predicted_lddt_loss, true_lddt, unclamped_fape,
             peptide_linker_scalers_list) = self.structure_module(
                single_representation=single_representation_tensor,
                pair_representation=pair_representation_tensor,
                sequence_amino_acid_labels=sequence_amino_acid_labels,
                ground_truth_transformation_matrix=current_cycle_input_batch["ground_truth_frames"],
                alternative_ground_truth_transformation_matrix=current_cycle_input_batch[
                    "alternative_ground_truth_frames"],
                ground_truth_angles=current_cycle_input_batch["ground_truth_angles"],
                alternative_ground_truth_angles=current_cycle_input_batch["alternative_ground_truth_angles"],
                ground_truth_positions=current_cycle_input_batch["ground_truth_global_positions"],
                alternative_ground_truth_positions=current_cycle_input_batch[
                    "alternative_ground_truth_global_positions"])

            previous_msa_representation_tensor = msa_representation_tensor
            previous_pair_representation_tensor = pair_representation_tensor
            previous_pseudo_carbon_beta_positions = pseudo_beta_positions

            model_outputs["angles"].append(angles)
            model_outputs["frames"].append(frames)
            model_outputs["final_positions"].append(final_positions)
            model_outputs["position_mask"].append(position_mask)
            model_outputs["pseudo_beta_positions"].append(pseudo_beta_positions)
            model_outputs["overall_fape_loss"].append(overall_fape_loss)
            model_outputs["auxillary_loss"].append(auxillary_loss)
            model_outputs["predicted_lddt_loss"].append(predicted_lddt_loss)
            model_outputs["true_lddt"].append(true_lddt)

            # todo to be reviewed
            if peptide_linker_scalers_list is not None:
                # stack the scalers across iterations: shape (number_iterations, batch, num_residues-1, 4)
                model_outputs["peptide_linker_scalers"].append(torch.stack(peptide_linker_scalers_list, dim=0))
            model_outputs["unclamped_fape"].append(unclamped_fape)


        # todo to be reviewed
        # Stack all tensors emanating from different cycles
        model_outputs = {key: torch.stack(value, dim=-1) if len(value) > 0 else None for key, value in model_outputs.items()}
        # Remove None values
        model_outputs = {k: v for k, v in model_outputs.items() if v is not None}

        # Compute distogram logits from the final pair representation and the distogram loss
        distogram_logits, distogram_probabilities = self.distogram_module(
            pair_representation=pair_representation_tensor)

        # Distogram labels are invariant and do not have a cycle dimension
        distogram_labels = batch_input_dictionary['distogram_labels']
        
        if self.enable_distogram_loss:
            distogram_loss = compute_distogram_loss(distogram_logits=distogram_logits, distogram_labels=distogram_labels)
        else:
            distogram_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        model_outputs["distogram_logits"] = distogram_logits
        model_outputs["distogram_loss"] = distogram_loss
        model_outputs["distogram_probabilities"] = distogram_probabilities

        return model_outputs
