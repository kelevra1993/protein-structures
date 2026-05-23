import torch
from pathlib import Path
from tests.utilities.testing_utilities import get_evoformer_test_inputs, test_nn_module_method
from architecture_modules.evoformer_module.evoformer import EvoformerBlock, EvoformerStack

config, test_inputs = get_evoformer_test_inputs()

# Unpack configuration variables
batch_size = config['batch_size']
msa_embedding = config['msa_embedding']
pair_representation_embedding = config['pair_representation_embedding']
head_embedding_dimension = config['head_embedding_dimension']
channel_scaler = config['channel_scaler']
intermediate_embedding = config['intermediate_embedding']
number_heads = config['number_heads']
triangle_multiplication_embedding = config['triangle_multiplication_embedding']

# Split simple and batched inputs
simple_inputs = {k: v[0] for k, v in test_inputs.items()}
batched_inputs = {k: v[1] for k, v in test_inputs.items()}

evoformer_stack_simple_inputs = {k: v[0].clone() for k, v in test_inputs.items()}
evoformer_stack_batched_inputs = {k: v[1].clone() for k, v in test_inputs.items()}

# 1. EvoformerBlock Test
evoformer_block = EvoformerBlock(
    msa_embedding=msa_embedding,
    pair_representation_embedding=pair_representation_embedding,
    msa_number_heads=8,
    msa_head_embedding_dimension=32,
    pair_number_heads=4,
    pair_head_embedding_dimension=32,
    channel_scaler=4,
    intermediate_embedding=32,
    triangle_multiplication_embedding=128,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=evoformer_block,
    input_tensor_dictionary=simple_inputs,
    output_tensor_names=["evo_block_m_out", "evo_block_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_inputs
)
print("EvoformerBlock Test Completed Successfuly.")

# 2. EvoformerStack Test
evoformer_stack = EvoformerStack(
    msa_embedding=msa_embedding,
    pair_representation_embedding=pair_representation_embedding,
    msa_number_heads=8,
    msa_head_embedding_dimension=32,
    pair_number_heads=4,
    pair_head_embedding_dimension=32,
    channel_scaler=4,
    intermediate_embedding=32,
    triangle_multiplication_embedding=128,
    number_blocks=3,
    single_representation_embedding=5,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=evoformer_stack,
    input_tensor_dictionary=evoformer_stack_simple_inputs,
    output_tensor_names=["evoformer_m_out", "evoformer_z_out", "evoformer_s_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=evoformer_stack_batched_inputs
)
print("EvoformerStack Test Completed Successfuly.")
