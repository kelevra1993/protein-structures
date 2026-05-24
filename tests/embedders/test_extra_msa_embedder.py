import math
import torch
from pathlib import Path

from embedders.extra_msa_embedder import ExtraMsaEmbedder, MSAColumnGlobalAttention, ExtraMsaBlock, ExtraMsaStack
from tests.utilities.testing_utilities import check_nn_module_method


def test_extra_msa_embedder_modules():
    # Known Reference Values
    batch_size = 3
    number_extra_sequences = 9
    number_residues = 35
    input_extra_msa_feature_dimension = 12
    extra_msa_embedding = 13
    msa_embedding_dimension = 4
    pair_representation_dimension = 5
    head_embedding_dimension = 6
    number_heads = 7
    number_clusters = 8

    # Shared Base Tensors
    input_shape = (number_extra_sequences, number_residues, input_extra_msa_feature_dimension)
    input_tensor = torch.linspace(-2, 2, math.prod(input_shape)).reshape(input_shape).double()

    msa_shape = (number_clusters, number_residues, msa_embedding_dimension)
    msa_tensor = torch.linspace(-2, 2, math.prod(msa_shape)).reshape(msa_shape).double()

    extra_msa_shape = (number_extra_sequences, number_residues, extra_msa_embedding)
    extra_msa_tensor = torch.linspace(-2, 2, math.prod(extra_msa_shape)).reshape(extra_msa_shape).double()

    pair_shape = (number_residues, number_residues, pair_representation_dimension)
    pair_tensor = torch.linspace(-2, 2, math.prod(pair_shape)).reshape(pair_shape).double()

    device = torch.device('cpu')
    dtype = torch.float64

    reference_folder = Path(__file__).parent / "extra_msa_embedder" / "reference_values"

    # 1. ExtraMsaEmbedder Test
    extra_msa_embedder = ExtraMsaEmbedder(
        input_extra_msa_feature_dimension=input_extra_msa_feature_dimension,
        extra_msa_embedding=extra_msa_embedding,
        device=device,
        dtype=dtype)

    check_nn_module_method(
        module=extra_msa_embedder,
        input_tensor_dictionary={'input_extra_msa_feature': input_tensor.clone()},
        output_tensor_names=['extra_msa_embedder_out'],
        reference_folder=reference_folder,
        batch_size=batch_size
    )
    print(" - ExtraMsaEmbedder Test Completed Successfuly.")

    # 2. MSAColumnGlobalAttention Test
    msa_column_global_attention = MSAColumnGlobalAttention(
        msa_embedding=msa_embedding_dimension,
        head_embedding_dimension=head_embedding_dimension,
        number_heads=number_heads,
        device=device,
        dtype=dtype)

    check_nn_module_method(
        module=msa_column_global_attention,
        input_tensor_dictionary={'msa_representation': msa_tensor.clone()},
        output_tensor_names=['msa_col_global_att_out'],
        reference_folder=reference_folder,
        batch_size=batch_size
    )
    print(" - MSAColumnGlobalAttention Test Completed Successfuly.")

    # 3. ExtraMsaBlock Test
    extra_msa_block = ExtraMsaBlock(
        extra_msa_embedding=extra_msa_embedding,
        pair_representation_embedding=pair_representation_dimension,
        device=device,
        dtype=dtype,
        msa_number_heads=8,
        msa_head_embedding_dimension=8,
        msa_global_number_heads=8,
        msa_global_head_embedding_dimension=8,
        pair_number_heads=4,
        pair_head_embedding_dimension=32,
        intermediate_embedding=32,
        channel_scaler=4,
        triangle_multiplication_embedding=128)

    check_nn_module_method(
        module=extra_msa_block,
        input_tensor_dictionary={
            'extra_msa_representation': extra_msa_tensor.clone(),
            'pair_representation': pair_tensor.clone()
        },
        output_tensor_names=[
            'extra_msa_block_msa_out',
            'extra_msa_block_pair_out'
        ],
        reference_folder=reference_folder,
        batch_size=batch_size)
    print(" - ExtraMsaBlock Test Completed Successfuly.")

    # 4. ExtraMsaStack Test
    extra_msa_stack = ExtraMsaStack(
        extra_msa_embedding=extra_msa_embedding,
        pair_representation_embedding=pair_representation_dimension,
        number_blocks=3,
        device=device,
        dtype=dtype,
        msa_number_heads=8,
        msa_head_embedding_dimension=8,
        msa_global_number_heads=8,
        msa_global_head_embedding_dimension=8,
        pair_number_heads=4,
        pair_head_embedding_dimension=32,
        intermediate_embedding=32,
        channel_scaler=4,
        triangle_multiplication_embedding=128)

    check_nn_module_method(
        module=extra_msa_stack,
        input_tensor_dictionary={
            'extra_msa_representation': extra_msa_tensor.clone(),
            'pair_representation': pair_tensor.clone()
        },
        output_tensor_names=['extra_msa_stack_out'],
        reference_folder=reference_folder,
        batch_size=batch_size
    )
    print(" - ExtraMsaStack Test Completed Successfuly.")
