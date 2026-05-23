import torch
from pathlib import Path
from tests.utilities.testing_utilities import get_evoformer_test_inputs, test_nn_module_method
from architecture_modules.evoformer_module.msa_stack import MSARowAttentionWithPairBias, MSAColumnAttention

config, test_inputs = get_evoformer_test_inputs()

# Unpack configuration variables
batch_size = config['batch_size']
msa_embedding = config['msa_embedding']
pair_representation_embedding = config['pair_representation_embedding']
number_heads = config['number_heads']
head_embedding_dimension = config['head_embedding_dimension']

# Split simple and batched inputs
simple_inputs = {k: v[0] for k, v in test_inputs.items()}
batched_inputs = {k: v[1] for k, v in test_inputs.items()}

# 1. MSARowAttentionWithPairBias
msa_row_attention = MSARowAttentionWithPairBias(
    msa_embedding=msa_embedding,
    pair_representation_embedding=pair_representation_embedding,
    head_embedding_dimension=head_embedding_dimension,
    number_heads=number_heads,
    device=torch.device('cpu'),
    dtype=torch.float64
)

print("MSA Stack Testing In Evoformer Module : ")

test_nn_module_method(
    module=msa_row_attention,
    input_tensor_dictionary=simple_inputs,
    output_tensor_names=["msa_row_att_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary=batched_inputs
)
print(" - MSARowAttentionWithPairBias Test Completed Successfuly.")

# 2. MSAColumnAttention
msa_column_attention = MSAColumnAttention(
    msa_embedding=msa_embedding,
    head_embedding_dimension=head_embedding_dimension,
    number_heads=number_heads,
    device=torch.device('cpu'),
    dtype=torch.float64
)

test_nn_module_method(
    module=msa_column_attention,
    input_tensor_dictionary={'msa_representation': simple_inputs['msa_representation']},
    output_tensor_names=["msa_col_att_out"],
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size,
    batched_input_tensor_dictionary={'msa_representation': batched_inputs['msa_representation']}
)
print(" - MSAColumnAttention Test Completed Successfuly.")
