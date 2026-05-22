import math
import torch
from torch import nn
from pathlib import Path

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention
from tests.utilities.testing_utilities import test_nn_module_method

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
    'attention_mask_tensor': (128,),
}


test_inputs = {
    key: torch.linspace(-2 - i / 5, 2 + i / 5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['attention_mask_tensor'] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0])
