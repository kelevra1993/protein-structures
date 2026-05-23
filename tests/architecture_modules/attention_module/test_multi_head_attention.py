import math
import torch
from torch import nn
from pathlib import Path

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention
from tests.utilities.testing_utilities import test_nn_module_method
from utilities.tensor_utilities import print_tensor_shape

batch_size = 2
input_dimension = 8
head_embedding_dimension = 10
number_heads = 4
attention_dimension = -3
dimension_1 = 3
dimension_2 = 5
dimension_3 = 6
dimension_4 = 7

feature_shapes = {
    'input_tensor': (dimension_1, dimension_2, dimension_3, dimension_4, input_dimension),
    'bias_tensor': (number_heads, dimension_3, dimension_3),
}

test_inputs = {
    key: torch.linspace(-2 - (i + 6) / 5, 2 + (i + 6) / 5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

# Test Configuration
test_configurations = {
    "gated": {
        "inputs": ["input_tensor"],
        "flags": {"use_gating": True, "use_global_attention": False, "use_embedding_bias": False}
    },
    "gated_bias": {
        "inputs": ["input_tensor", "bias_tensor"],
        "flags": {"use_gating": True, "use_global_attention": False, "use_embedding_bias": False}
    },
    "ungated": {
        "inputs": ["input_tensor"],
        "flags": {"use_gating": False, "use_global_attention": False, "use_embedding_bias": False}
    },
    "global": {
        "inputs": ["input_tensor"],
        "flags": {"use_gating": False, "use_global_attention": True, "use_embedding_bias": False}
    }
}

for test_name, config in test_configurations.items():
    # Instantiate the module with the specific configuration flags
    multi_head_attention = MultiHeadAttention(
        input_dimension=input_dimension,
        head_embedding_dimension=head_embedding_dimension,
        number_heads=number_heads,
        attention_dimension=attention_dimension,
        device=torch.device('cpu'),
        dtype=torch.float64,
        **config["flags"]
    )

    # Filter test_inputs to only pass the required arguments for this case
    inputs = {input_name: test_inputs[input_name] for input_name in config["inputs"]}

    test_nn_module_method(
        module=multi_head_attention,
        input_tensor_dictionary=inputs,
        output_tensor_names=[test_name],
        reference_folder=Path(__file__).parent / "reference_values",
        batch_size=batch_size
    )
    print(f"MultiHeadAttention Test For '{test_name}' Completed Successfuly!")
