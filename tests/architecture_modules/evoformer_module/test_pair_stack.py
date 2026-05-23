import torch
from pathlib import Path
from tests.utilities.testing_utilities import get_evoformer_test_inputs, test_nn_module_method
from architecture_modules.evoformer_module.pair_stack import (
    TriangleMultiplication, TriangleAttention, PairTransition, PairStack
)

config, test_inputs = get_evoformer_test_inputs()

# Unpack configuration variables
batch_size = config['batch_size']
pair_representation_embedding = config['pair_representation_embedding']
head_embedding_dimension = config['head_embedding_dimension']
number_heads = config['number_heads']
channel_scaler = config['channel_scaler']
triangle_multiplication_embedding = config['triangle_multiplication_embedding']
embedding_dimension = head_embedding_dimension

# Split simple and batched inputs
simple_inputs = {k: v[0] for k, v in test_inputs.items()}
batched_inputs = {k: v[1] for k, v in test_inputs.items()}

# We only need the pair_representation for the Pair Stack modules
simple_pair_input = {'pair_representation': simple_inputs['pair_representation']}
batched_pair_input = {'pair_representation': batched_inputs['pair_representation']}


# 1. TriangleMultiplication (Outgoing)
triangle_multiplication_outgoing = TriangleMultiplication(
    pair_representation_embedding=pair_representation_embedding,
    multiplication_type="outgoing",
    embedding_dimension=embedding_dimension,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=triangle_multiplication_outgoing,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["tri_mul_out_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - TriangleMultiplication (Outgoing) Test Completed Successfuly.")

# 2. TriangleMultiplication (Incoming)
triangle_multiplication_incoming = TriangleMultiplication(
    pair_representation_embedding=pair_representation_embedding,
    multiplication_type="incoming",
    embedding_dimension=embedding_dimension,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=triangle_multiplication_incoming,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["tri_mul_in_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - TriangleMultiplication (Incoming) Test Completed Successfuly.")


# 3. TriangleAttention (Starting Node)
triangle_attention_starting = TriangleAttention(
    pair_representation_embedding=pair_representation_embedding,
    node_type="starting_node",
    head_embedding_dimension=head_embedding_dimension,
    number_heads=number_heads,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=triangle_attention_starting,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["tri_att_start_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - TriangleAttention (Starting Node) Test Completed Successfuly.")

# 4. TriangleAttention (Ending Node)
triangle_attention_ending = TriangleAttention(
    pair_representation_embedding=pair_representation_embedding,
    node_type="ending_node",
    head_embedding_dimension=head_embedding_dimension,
    number_heads=number_heads,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=triangle_attention_ending,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["tri_att_end_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - TriangleAttention (Ending Node) Test Completed Successfuly.")


# 5. PairTransition
pair_transition = PairTransition(
    pair_representation_embedding=pair_representation_embedding,
    channel_scaler=channel_scaler,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=pair_transition,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["pair_transition_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - PairTransition Test Completed Successfuly.")


# 6. PairStack
pair_stack = PairStack(
    pair_representation_dimension=pair_representation_embedding,
    head_embedding_dimension=32,
    triangle_multiplication_embedding=triangle_multiplication_embedding,
    number_heads=4,
    channel_scaler=4,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=pair_stack,
    input_tensor_dictionary=simple_pair_input,
    output_tensor_names=["pair_stack_z_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_pair_input
)
print(" - PairStack Test Completed Successfuly.")
