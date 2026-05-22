import torch
import numpy as np

from typing import Optional
from torch import nn


class MultiHeadAttention(nn.Module):
    """"""

    def __init__(self, input_dimension: int, head_embedding_dimension: int, number_heads: int, attention_dimension: int,
                 use_gating: bool,
                 use_global_attention: bool, use_embedding_bias: bool, device: torch.device, dtype: torch.dtype):
        """"""

        super().__init__()

        self.input_dimension = input_dimension
        self.head_embedding_dimension = head_embedding_dimension
        self.number_heads = number_heads
        self.use_gating = use_gating
        self.attention_dimension = attention_dimension
        self.use_global_attention = use_global_attention
        self.device = device
        self.dtype = dtype

        # Note to be added in the docstring that query, key and value embedders are subject to usage of bias or not.

        self.query_embedder = nn.Linear(in_features=input_dimension,
                                        out_features=number_heads * head_embedding_dimension,
                                        bias=use_embedding_bias, dtype=self.dtype, device=self.device)

        number_key_value_heads = 1 if self.use_global_attention else number_heads
        self.key_embedder = nn.Linear(in_features=input_dimension,
                                      out_features=number_key_value_heads * head_embedding_dimension,
                                      bias=use_embedding_bias, dtype=self.dtype, device=self.device)

        self.value_embedder = nn.Linear(in_features=input_dimension,
                                        out_features=number_key_value_heads * head_embedding_dimension,
                                        bias=use_embedding_bias, dtype=self.dtype, device=self.device)

        self.output_embedder = nn.Linear(in_features=number_heads * head_embedding_dimension,
                                         out_features=input_dimension,
                                         dtype=self.dtype, device=self.device)

        # Choice To Use Gating That Will Scale The (Attention * Value) Outputs With An Element-Wise Multiplication
        if self.use_gating:
            self.gating_embedder = nn.Linear(in_features=input_dimension,
                                             out_features=number_heads * head_embedding_dimension,
                                             dtype=self.dtype, device=self.device)

    def separate_key_query_value_heads(self, query_embedding: torch.Tensor, key_embedding: torch.Tensor,
                                       value_embedding: torch.Tensor):
        """"""

        # Move Dimensions To Accomodate For Attention Dimension
        query_embedding = query_embedding.movedim(source=self.attention_dimension, destination=-2)
        key_embedding = key_embedding.movedim(source=self.attention_dimension, destination=-2)
        value_embedding = value_embedding.movedim(source=self.attention_dimension, destination=-2)

        # Split Along The Last Dimension
        query_embedding_chunks = query_embedding.split(split_size=self.head_embedding_dimension, dim=-1)
        key_embedding_chunks = key_embedding.split(split_size=self.head_embedding_dimension, dim=-1)
        value_embedding_chunks = value_embedding.split(split_size=self.head_embedding_dimension, dim=-1)

        # Re-Stack The Embeddings
        query_embedding = torch.stack(tensors=query_embedding_chunks, dim=-3)
        key_embedding = torch.stack(key_embedding_chunks, dim=-3)
        value_embedding = torch.stack(value_embedding_chunks, dim=-3)

        if self.use_global_attention:
            query_embedding = torch.mean(query_embedding, dim=-2, keepdim=True)

        return query_embedding, key_embedding, value_embedding

    def forward(self, input_tensor: torch.Tensor, bias_tensor=None, attention_mask_tensor=None):
        """"""
        query_embedding = self.query_embedder(input_tensor)
        key_embedding = self.key_embedder(input_tensor)
        value_embedding = self.value_embedder(input_tensor)

        query_embedding, key_embedding, value_embedding = self.separate_key_query_value_heads(
            query_embedding=query_embedding, key_embedding=key_embedding, value_embedding=value_embedding)

        # Compute Attention Tensor And Scale It Accordignly
        attention_tensor = torch.matmul(
            input=query_embedding,
            other=torch.transpose(key_embedding, dim0=-1, dim1=-2)) / np.sqrt(self.head_embedding_dimension)

        # Add Bias If Present : We Might Have To Reshape It accordignly
        if bias_tensor is not None:
            bias_batch_shape = bias_tensor.shape[:-3]
            pre_broad_cast_shape = (bias_batch_shape + (1,) *
                                    (attention_tensor.ndim - len(bias_batch_shape) - 3) +
                                    bias_tensor.shape[-3:])

            bias_tensor = bias_tensor.reshape(pre_broad_cast_shape)
            attention_tensor += bias_tensor

        # Will Be Useful For When We Will Be Batching Multiple Different Proteins
        if attention_mask_tensor is not None:

            for squeeze_dimension in [-2, -3]:
                attention_mask_tensor = torch.unsqueeze(attention_mask_tensor, dim=squeeze_dimension)

            # Hard offset equivalent to minus infinity turning softmax to effectivly 0 for these values.
            offset = -1e8 * (attention_mask_tensor == 0).to(torch.float)
            attention_tensor += offset

        # Softmax For Attention Tensor
        # Scaled by the square root of head_dimension similar to what we did for image classification,
        # when we did not want a skewed output for our prediction probabilities
        attention_tensor = torch.softmax(attention_tensor, dim=-1)
        attentioned_value_embedding = torch.matmul(input=attention_tensor, other=value_embedding)

        # Re concatenated heads
        attentioned_value_embedding = attentioned_value_embedding.movedim(source=-3, destination=-2)
        attentioned_value_embedding = torch.flatten(attentioned_value_embedding, start_dim=-2)
        attentioned_value_embedding = attentioned_value_embedding.movedim(source=-2,
                                                                          destination=self.attention_dimension)

        if self.use_gating:
            sigmoid_gate_embedding = torch.sigmoid(self.gating_embedder(input_tensor))
            attentioned_value_embedding = attentioned_value_embedding * sigmoid_gate_embedding

        # Run The Output Through The Final Embedder
        output_tensor = self.output_embedder(attentioned_value_embedding)

        return output_tensor
