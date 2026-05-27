

import torch
from torch import nn

from embedders.input_embedder import InputEmbedder
from embedders.recycling_embedder import RecyclingEmbedder
from embedders.extra_msa_embedder import ExtraMsaStack, ExtraMsaEmbedder
from architecture_modules.evoformer_module.evoformer import EvoformerStack
from architecture_modules.structure_module.structure_module import StructureModule


class Model(nn.Module):

    def __init__(self, input_sequence_feature_dimension: int = 21, input_msa_feature_dimension: int = 49,
                 input_extra_msa_feature_dimension: int = 25, number_neighbouring_amino_acids: int = 32,
                 single_representation_embedding: int = 384, pair_representation_embedding: int = 128,
                 msa_embedding: int = 256, extra_msa_embedding: int = 64,
                 number_extra_msa_blocks: int = 4, number_evoformer_blocks: int = 48,
                 extra_msa_stack_msa_number_heads: int = None, extra_msa_stack_msa_head_embedding_dimension: int = None,
                 extra_msa_stack_msa_global_number_heads: int = None, extra_msa_stack_msa_global_head_embedding_dimension: int = None,
                 extra_msa_stack_pair_number_heads: int = None, extra_msa_stack_pair_head_embedding_dimension: int = None,
                 extra_msa_stack_intermediate_embedding: int = None, extra_msa_stack_channel_scaler: int = None,
                 extra_msa_stack_triangle_multiplication_embedding: int = None,
                 evoformer_number_heads: int = None, evoformer_head_embedding_dimension: int = None,
                 structure_module_number_heads: int = None, structure_module_head_embedding_dimension: int = None,
                 number_query_points: int = None, number_value_points: int = None,
                 number_structure_module_iterations: int = None, number_torsion_angles: int = 7,
                 device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the Alphafold model.

        Args:
            msa_embedding (int, optional): Number of channels for the MSA representation. Defaults to 256.
            pair_representation_embedding (int, optional): Number of channels for the pair representation. Defaults to 128.
            extra_msa_embedding (int, optional): Number of channels for the extra MSA representation. Defaults to 64.
            input_extra_msa_feature_dimension (int, optional): Number of channels of the extra MSA feature. Defaults to 25.
            input_sequence_feature_dimension (int, optional): Number of channels of the target feature. Defaults to 22.
            input_msa_feature_dimension (int, optional): Dimension of input MSA features. Defaults to 49.
            number_neighbouring_amino_acids (int, optional): Window size for relative position encoding. Defaults to None.
            single_representation_embedding (int, optional): Number of channels for the single representation. Defaults to 384.
            number_extra_msa_blocks (int, optional): Number of blocks for the extra MSA stack. Defaults to 4.
            number_evoformer_blocks (int, optional): Number of blocks for the Evoformer. Defaults to 48.
            extra_msa_stack_msa_number_heads (int, optional): Number of attention heads for the MSA row-wise attention. Defaults to None.
            extra_msa_stack_msa_head_embedding_dimension (int, optional): Dimension per attention head for the MSA row-wise attention. Defaults to None.
            extra_msa_stack_msa_global_number_heads (int, optional): Number of attention heads for the MSA global column-wise attention. Defaults to None.
            extra_msa_stack_msa_global_head_embedding_dimension (int, optional): Dimension per attention head for the MSA global column-wise attention. Defaults to None.
            extra_msa_stack_pair_number_heads (int, optional): Number of attention heads for the pair stack embedder. Defaults to None.
            extra_msa_stack_pair_head_embedding_dimension (int, optional): Dimension per attention head for the pair stack embedder. Defaults to None.
            extra_msa_stack_intermediate_embedding (int, optional): Intermediate dimension for OuterProductMean in Extra MSA stack. Defaults to None.
            extra_msa_stack_channel_scaler (int, optional): Channel scaler for transition layers in Extra MSA stack. Defaults to None.
            extra_msa_stack_triangle_multiplication_embedding (int, optional): Dimension for triangle multiplication in Extra MSA stack. Defaults to None.
            evoformer_number_heads (int, optional): Number of attention heads for the Evoformer. Defaults to None.
            evoformer_head_embedding_dimension (int, optional): Dimension of each attention head for the Evoformer. Defaults to None.
            structure_module_number_heads (int, optional): Number of attention heads for the Structure Module. Defaults to None.
            structure_module_head_embedding_dimension (int, optional): Dimension of each attention head for the Structure Module. Defaults to None.
            number_query_points (int, optional): Number of query points. Defaults to None.
            number_value_points (int, optional): Number of value points. Defaults to None.
            # update this number_structure_module_iterations (int, optional): Number of layers in the Structure Module. Defaults to None.
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
        self.input_msa_feature_dimension = input_msa_feature_dimension
        self.number_neighbouring_amino_acids = number_neighbouring_amino_acids
        self.number_extra_msa_blocks = number_extra_msa_blocks
        self.number_evoformer_blocks = number_evoformer_blocks
        self.extra_msa_stack_msa_number_heads = extra_msa_stack_msa_number_heads
        self.extra_msa_stack_msa_head_embedding_dimension = extra_msa_stack_msa_head_embedding_dimension
        self.extra_msa_stack_msa_global_number_heads = extra_msa_stack_msa_global_number_heads
        self.extra_msa_stack_msa_global_head_embedding_dimension = extra_msa_stack_msa_global_head_embedding_dimension
        self.extra_msa_stack_pair_number_heads = extra_msa_stack_pair_number_heads
        self.extra_msa_stack_pair_head_embedding_dimension = extra_msa_stack_pair_head_embedding_dimension
        self.extra_msa_stack_intermediate_embedding = extra_msa_stack_intermediate_embedding
        self.extra_msa_stack_channel_scaler = extra_msa_stack_channel_scaler
        self.extra_msa_stack_triangle_multiplication_embedding = extra_msa_stack_triangle_multiplication_embedding
        self.evoformer_number_heads = evoformer_number_heads
        self.evoformer_head_embedding_dimension = evoformer_head_embedding_dimension
        self.structure_module_number_heads = structure_module_number_heads
        self.structure_module_head_embedding_dimension = structure_module_head_embedding_dimension
        self.number_query_points = number_query_points
        self.number_value_points = number_value_points
        self.number_structure_module_iterations = number_structure_module_iterations
        self.number_torsion_angles = number_torsion_angles
        self.device = device
        self.dtype = dtype

        self.input_embedder = InputEmbedder(
            input_sequence_feature_dimension=self.input_sequence_feature_dimension,
            input_msa_feature_dimension=self.input_msa_feature_dimension,
            input_extra_msa_feature_dimension=self.input_extra_msa_feature_dimension,
            msa_embedding=self.msa_embedding,
            extra_msa_embedding=self.extra_msa_embedding,
            pair_representation_embedding=self.pair_representation_embedding,
            number_neighbouring_amino_acids=self.number_neighbouring_amino_acids,
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
            msa_number_heads=self.extra_msa_stack_msa_number_heads,
            msa_head_embedding_dimension=self.extra_msa_stack_msa_head_embedding_dimension,
            msa_global_number_heads=self.extra_msa_stack_msa_global_number_heads,
            msa_global_head_embedding_dimension=self.extra_msa_stack_msa_global_head_embedding_dimension,
            pair_number_heads=self.extra_msa_stack_pair_number_heads,
            pair_head_embedding_dimension=self.extra_msa_stack_pair_head_embedding_dimension,
            intermediate_embedding=self.extra_msa_stack_intermediate_embedding,
            channel_scaler=self.extra_msa_stack_channel_scaler,
            triangle_multiplication_embedding=self.extra_msa_stack_triangle_multiplication_embedding,
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
            number_iterations=self.number_structure_module_iterations,
            number_torsion_angles=self.number_torsion_angles,
            device=self.device,
            dtype=self.dtype
        )

