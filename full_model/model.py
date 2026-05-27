

import torch
from torch import nn

from embedders.input_embedder import InputEmbedder
from embedders.recycling_embedder import RecyclingEmbedder
from embedders.extra_msa_embedder import ExtraMsaStack, ExtraMsaEmbedder
from architecture_modules.evoformer_module.evoformer import EvoformerStack
from architecture_modules.structure_module.structure_module import StructureModule


class Model(nn.Module):

    def __init__(self, msa_embedding=256, pair_representation_embedding=128, extra_msa_embedding=64, 
                 input_extra_msa_feature_dimension=25, input_sequence_feature_dimension=21, 
                 single_representation_embedding=384, number_extra_msa_blocks=4,
                 number_evoformer_blocks=48, 
                 extra_msa_number_heads=None, extra_msa_head_embedding_dimension=None,
                 evoformer_number_heads=None, evoformer_head_embedding_dimension=None,
                 structure_module_number_heads=None, structure_module_head_embedding_dimension=None,
                 number_query_points=None, number_value_points=None,
                 number_layers=None, number_torsion_angles=None,
                 device=None, dtype=None):
        """
        Initializes the Alphafold model.

        Args:
            msa_embedding (int, optional): Number of channels for the MSA representation. Defaults to 256.
            pair_representation_embedding (int, optional): Number of channels for the pair representation. Defaults to 128.
            extra_msa_embedding (int, optional): Number of channels for the extra MSA representation. Defaults to 64.
            input_extra_msa_feature_dimension (int, optional): Number of channels of the extra MSA feature. Defaults to 25.
            input_sequence_feature_dimension (int, optional): Number of channels of the target feature. Defaults to 22.
            single_representation_embedding (int, optional): Number of channels for the single representation. Defaults to 384.
            number_extra_msa_blocks (int, optional): Number of blocks for the extra MSA stack. Defaults to 4.
            number_evoformer_blocks (int, optional): Number of blocks for the Evoformer. Defaults to 48.
            extra_msa_number_heads (int, optional): Number of attention heads for the extra MSA stack. Defaults to None.
            extra_msa_head_embedding_dimension (int, optional): Dimension of each attention head for the extra MSA stack. Defaults to None.
            evoformer_number_heads (int, optional): Number of attention heads for the Evoformer. Defaults to None.
            evoformer_head_embedding_dimension (int, optional): Dimension of each attention head for the Evoformer. Defaults to None.
            structure_module_number_heads (int, optional): Number of attention heads for the Structure Module. Defaults to None.
            structure_module_head_embedding_dimension (int, optional): Dimension of each attention head for the Structure Module. Defaults to None.
            number_query_points (int, optional): Number of query points. Defaults to None.
            number_value_points (int, optional): Number of value points. Defaults to None.
            number_layers (int, optional): Number of layers in the Structure Module. Defaults to None.
            number_torsion_angles (int, optional): Number of torsion angles to predict. Defaults to None.
            device (torch.device, optional): Device on which the model should run.
            dtype (torch.dtype, optional): Data type of the model.
        """
        super().__init__()

        self.msa_embedding = msa_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.extra_msa_embedding = extra_msa_embedding
        self.single_representation_embedding = single_representation_embedding
        self.input_extra_msa_feature_dimension = input_extra_msa_feature_dimension
        self.input_sequence_feature_dimension = input_sequence_feature_dimension
        self.number_extra_msa_blocks = number_extra_msa_blocks
        self.number_evoformer_blocks = number_evoformer_blocks
        self.extra_msa_number_heads = extra_msa_number_heads
        self.extra_msa_head_embedding_dimension = extra_msa_head_embedding_dimension
        self.evoformer_number_heads = evoformer_number_heads
        self.evoformer_head_embedding_dimension = evoformer_head_embedding_dimension
        self.structure_module_number_heads = structure_module_number_heads
        self.structure_module_head_embedding_dimension = structure_module_head_embedding_dimension
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_layers = number_layers
        self.number_torsion_angles = number_torsion_angles
        self.device = device
        self.dtype = dtype

        self.input_embedder = InputEmbedder(
            msa_embedding=self.msa_embedding, 
            pair_representation_embedding=self.pair_representation_embedding, 
            input_sequence_feature_dimension=self.input_sequence_feature_dimension,
            device=self.device,
            dtype=self.dtype
        )
        self.extra_msa_embedder = ExtraMsaEmbedder(
            input_extra_msa_feature_dimension=self.input_extra_msa_feature_dimension, 
            extra_msa_embedding=self.extra_msa_embedding,
            device=self.device,
            dtype=self.dtype
        )
        self.recycling_embedder = RecyclingEmbedder(
            msa_embedding=self.msa_embedding, 
            pair_representation_embedding=self.pair_representation_embedding,
            device=self.device,
            dtype=self.dtype
        )
        self.extra_msa_stack = ExtraMsaStack(
            extra_msa_embedding=self.extra_msa_embedding, 
            pair_representation_embedding=self.pair_representation_embedding, 
            number_blocks=self.number_extra_msa_blocks,
            number_heads=self.extra_msa_number_heads,
            head_embedding_dimension=self.extra_msa_head_embedding_dimension,
            device=self.device,
            dtype=self.dtype
        )
        self.evoformer = EvoformerStack(
            msa_embedding=self.msa_embedding, 
            pair_representation_embedding=self.pair_representation_embedding, 
            number_blocks=self.number_evoformer_blocks,
            number_heads=self.evoformer_number_heads,
            head_embedding_dimension=self.evoformer_head_embedding_dimension,
            device=self.device,
            dtype=self.dtype
        )
        self.structure_module = StructureModule(
            single_representation_embedding=self.single_representation_embedding, 
            pair_representation_embedding=self.pair_representation_embedding,
            number_heads=self.structure_module_number_heads,
            head_embedding_dimension=self.structure_module_head_embedding_dimension,
            number_query_points=self.number_query_points,
            number_value_points=self.number_value_points,
            number_layers=self.number_layers,
            number_torsion_angles=self.number_torsion_angles,
            device=self.device,
            dtype=self.dtype
        )

