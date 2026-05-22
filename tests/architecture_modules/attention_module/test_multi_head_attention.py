import math
import torch
from torch import nn

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
    'x': (dimension_1, dimension_2, dimension_3, dimension_4, input_dimension),
    'bias': (number_heads, dimension_3, dimension_3),
    'fake_attention_mask': (dimension_1, dimension_2, dimension_4, dimension_3),
    'attention_mask': (128,),
}

batched_feature_shapes = {
    key: (batch_size,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2 - i / 5, 2 + i / 5, math.prod(shape)).reshape(shape).double()
    for i, (key, shape) in enumerate(feature_shapes.items())
}

test_inputs['fake_attention_mask'] = torch.ones((dimension_1, dimension_2, dimension_4, dimension_3))
test_inputs['fake_attention_mask'][..., 2:4] = 0

test_inputs['input_ids'] = torch.tensor([101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0])

test_inputs['attention_mask'] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0])
