

import torch
from torch import nn

from embedders.input_embedder import InputEmbedder
from embedders.recycling_embedder import RecyclingEmbedder
from embedders.extra_msa_embedder import ExtraMsaStack, ExtraMsaEmbedder
from architecture_modules.evoformer_module.evoformer import EvoformerStack
from architecture_modules.structure_module.structure_module import StructureModule


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

        # TODO Kept for usage in the forward pass
        self.msa_embedding = global_configuration.get('msa_embedding')
        self.pair_representation_embedding = global_configuration.get('pair_representation_embedding')
        self.extra_msa_embedding = global_configuration.get('extra_msa_embedding')
        self.single_representation_embedding = global_configuration.get('single_representation_embedding')

        self.input_embedder = InputEmbedder(
            input_sequence_feature_dimension=global_configuration.get('input_sequence_feature_dimension'),
            input_msa_feature_dimension=global_configuration.get('input_msa_feature_dimension'),
            input_extra_msa_feature_dimension=global_configuration.get('input_extra_msa_feature_dimension'),
            msa_embedding=global_configuration.get('msa_embedding'),
            extra_msa_embedding=global_configuration.get('extra_msa_embedding'),
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
            msa_global_head_embedding_dimension=extra_msa_stack_configuration.get('msa_global_head_embedding_dimension'),
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
            angle_representation_embedding=global_configuration.get('single_representation_embedding'),
            number_query_points=structure_module_configuration.get('number_query_points'),
            number_value_points=structure_module_configuration.get('number_value_points'),
            number_heads=structure_module_configuration.get('number_heads'),
            head_embedding_dimension=structure_module_configuration.get('head_embedding_dimension'),
            number_torsion_angles=structure_module_configuration.get('number_torsion_angles'),
            device=self.device,
            dtype=self.dtype)






    """
    Note for documentation : 
    here the forward expects a dictionary that looks like this :
    batch={
        'msa_feat': msa_feat,
        'extra_msa_feat': extra_msa_feat,
        'target_feat': target_feat,
        'residue_index': residue_index
    }
    but in our code,
    we change the feature extractor so that it can output something like this :
    input_sequence_feature (this corresponds to target_feat)
    input_residue_index_feature (this corresponds to residue_index)
    input_msa_feature (this corresponds to msa_feat)
    input_extra_msa_feature (this corresponds to extra_msa_feat)
    so change the keys in the batch dictionary accordingly.
    """

    def forward(self, batch):
        """
        Forward pass for the Alphafold model.

        Args:
            batch (dict): A dictionary containing the following features:
                * msa_feat:  Tensor of shape (*, N_seq, N_res, msa_feat_dim, N_cycle).
                * extra_msa_feat: Tensor of shape (*, N_extra, N_res, f_e, N_cycle).
                * target_feat: Tensor of shape (*, N_res, tf_dim, N_cycle). One-hot encoding of the target sequence.
                * residue_index: Tensor of shape (*, N_res, N_cycle). The index of each residue, which is [0,...,N_res-1].

        Returns:
            dict: A dictionary with the following entries:
                * final_positions: Heavy-atom positions in Angstrom of shape (*, N_res, 37, 3, N_cycle).
                * position_mask: Boolean tensor of shape (*, N_res, 37, N_cycle), masking atoms that
                    aren't present in the amino acids.
                * angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2, N_cycle) for
                    every iteration of the Structure Module in every cycle.
                * frames: Backbone frames of shape (*, N_layers, N_res, 4, 4, N_cycle) for every iteration
                    of the Structure Module in every cycle.
        """
        N_cycle = batch['msa_feat'].shape[-1]
        N_seq, N_res = batch['msa_feat'].shape[-4:-2]
        batch_shape = batch['msa_feat'].shape[:-4]
        device = batch['msa_feat'].device
        dtype = batch['msa_feat'].dtype

        c_m = self.c_m
        c_z = self.c_z

        outputs = {}

        # todo will have to put dtype in every single class as an input and by default dtype=torch.float64
        # Todo Remind Yourself that N_seq is for the msa sequences and does not represent the batch
        # Todo condsider calling .forward explicitly to explicitly hand out arguments

        # Initialisation of first tensors
        prev_m = torch.zeros((batch_shape + (N_seq, N_res, c_m)), dtype=torch.float64)
        prev_z = torch.zeros((batch_shape + (N_res, N_res, c_z)), dtype=torch.float64)
        prev_pseudo_beta_x = torch.zeros((batch_shape + (N_res, 3)), dtype=torch.float64)

        for cycle in range(N_cycle):
            print(20 * '-')
            print(f'Starting iteration {cycle}')

            current_cycle_input_batch = {key: value[..., cycle] for key, value in batch.items()}

            # Here current_cycle_input_batch has to be totally unpacked so that it can be fed to the
            # self.input_embedder explicity while showing the arguments func(arg1=variable1,....)
            msa_tensor, pair_rep_tensor = self.input_embedder(current_cycle_input_batch)

            # TODO Same here for the arguments in the calling of the function
            recycled_msa, recycled_pair_rep = self.recycling_embedder(prev_m, prev_z, prev_pseudo_beta_x)

            # The very first sequence of the msa
            # The recycling embedder just actually normalizes the msa_representation_input using a layer normaliser
            # todo just to check that the first recycled_msa is actually 0
            msa_tensor[..., 0, :, :] += recycled_msa
            pair_rep_tensor += recycled_pair_rep

            # TODO Same here for the arguments in the calling of the function
            # Here this is just a simple linear layer
            extra_msa_embedding = self.extra_msa_embedder(current_cycle_input_batch)

            # TODO Same here for the arguments in the calling of the function
            # The pair representation is updated by the extra msa embedding before being fed to the evoformer stack
            pair_rep_tensor = self.extra_msa_stack(extra_msa_embedding, pair_rep_tensor)

            # Pass through the evorformer block
            # TODO Same here for the arguments in the calling of the function
            msa_tensor, pair_rep_tensor, single_representation_tensor = self.evoformer(msa_tensor, pair_rep_tensor)

            # TODO Rename F not clear enough
            F = torch.argmax(current_cycle_input_batch["target_feat"], dim=-1)

            # TODO Same here for the arguments in the calling of the function
            # For structure module we actually have a certain amount of outputs that are not dictionaries so change the output to reflect that
            structure_module_output = self.structure_module(single_representation_tensor, pair_rep_tensor, F)

            prev_m = msa_tensor
            prev_z = pair_rep_tensor
            prev_pseudo_beta_x = structure_module_output['pseudo_beta_positions']

            # todo to be changed since the output of the structure module is not a dictionary
            for key, value in structure_module_output.items():
                if key in outputs:
                    outputs[key].append(value)
                else:
                    outputs[key] = [value]

        outputs = {
            key: torch.stack(value, dim=-1) for key, value in outputs.items()
        }

        return outputs


if __name__ == "__main__":
    import os
    from pathlib import Path
    from utilities.configuration_utilities import load_configuration
    from utilities.tensor_utilities import get_device

    project_folder = Path(os.getcwd()).parent
    configuration_file = project_folder / "configurations" / "template_configuration.yaml"

    model_configuration = load_configuration(configuration_path=configuration_file)
    computer_device = get_device()

    if str(computer_device) == "mps":
        tensor_dtype = torch.float32
    else:
        tensor_dtype = torch.float64

    # Test the alphafold model with template configuration
    alphafold_model = Model(configuration=model_configuration, device=computer_device, dtype=tensor_dtype)
