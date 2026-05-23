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

number_blocks = None
single_representation_embedding = None
################################################################################

# Split simple and batched inputs
simple_inputs = {k: v[0] for k, v in test_inputs.items()}
batched_inputs = {k: v[1] for k, v in test_inputs.items()}

# 1. EvoformerBlock Test
evoformer_block = EvoformerBlock(
    msa_embedding=msa_embedding,
    pair_representation_embedding=pair_representation_embedding,
    number_heads=8,
    head_embedding_dimension=32,
    channel_scaler=channel_scaler,
    intermediate_embedding=intermediate_embedding,
    triangle_multiplication_embedding=triangle_multiplication_embedding,
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
exit()
# 2. EvoformerStack Test
evoformer_stack = EvoformerStack(
    msa_embedding=msa_embedding,
    pair_representation_embedding=pair_representation_embedding,
    number_heads=number_heads,
    head_embedding_dimension=head_embedding_dimension,
    channel_scaler=channel_scaler,
    intermediate_embedding=intermediate_embedding,
    triangle_multiplication_embedding=triangle_multiplication_embedding,
    number_blocks=number_blocks,
    single_representation_embedding=single_representation_embedding,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=evoformer_stack,
    input_tensor_dictionary=simple_inputs,
    output_tensor_names=["evoformer_m_out", "evoformer_z_out", "evoformer_s_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_inputs
)
print("EvoformerStack Test Completed Successfuly.")
