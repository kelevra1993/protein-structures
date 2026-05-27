import torch
import numpy as np

from typing import Optional, Tuple
from torch import nn


class MultiHeadAttention(nn.Module):
    """"""

    def __init__(self, input_dimension: int, head_embedding_dimension: int, number_heads: int, attention_dimension: int,
                 use_gating: bool,
                 use_global_attention: bool, use_embedding_bias: bool, device: torch.device, dtype: torch.dtype):
        """
        Initializes the MultiHeadAttention module, a core component of the AlphaFold II architecture.

        This module is used for various attention mechanisms within the model, such as Row-wise or Column-wise
        Gated Self-Attention in the Evoformer blocks, and for processing information in the Structure Module.
        It computes the interactions between residues by projecting inputs into Query, Key, and Value spaces,
        optionally applying gating and global attention mechanisms.

        Args:
            input_dimension: The size of the last dimension of the input tensor (e.g., msa_embedding_dimension
                or pair_representation_dimension).
            head_embedding_dimension: The dimension of each individual attention head.
            number_heads: The total number of attention heads to use.
            attention_dimension: The index of the dimension along which attention is computed (the residue axis).
            use_gating: If True, applies a sigmoid gating mechanism to the attention output to control information flow.
            use_global_attention: If True, computes a global representation by averaging queries, effectively
                reducing the complexity for certain architectural components.
            use_embedding_bias: If True, enables bias in the linear projections for Query, Key, and Value.
            device: The torch device on which the module's parameters will be allocated.
            dtype: The torch data type for the module's parameters.
        """

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
                                       value_embedding: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the projected Query, Key, and Value embeddings into multiple attention heads and
        reshapes them for the attention computation.

        This method moves the residue dimension to a standard position and splits the embedding
        dimension into the specified number of heads. It also handles the 'global attention'
        case where queries are averaged across the residue dimension.

        Args:
            query_embedding: Projected query tensor of shape
                (..., number_residues, number_heads * head_embedding_dimension).
            key_embedding: Projected key tensor of shape
                (..., number_residues, number_key_value_heads * head_embedding_dimension).
            value_embedding: Projected value tensor of shape
                (..., number_residues, number_key_value_heads * head_embedding_dimension).

        Returns:
            A tuple of (query_embedding, key_embedding, value_embedding) where:
                - query_embedding: Reshaped query tensor of shape
                  (..., number_heads, number_residues, head_embedding_dimension)
                  (or (..., number_heads, 1, head_embedding_dimension) if use_global_attention is True).
                - key_embedding: Reshaped key tensor of shape
                  (..., number_key_value_heads, number_residues, head_embedding_dimension).
                - value_embedding: Reshaped value tensor of shape
                  (..., number_key_value_heads, number_residues, head_embedding_dimension).
        """

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

    def forward(self, input_tensor: torch.Tensor, bias_tensor: Optional[torch.Tensor] = None,
                attention_mask_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the MultiHeadAttention module.

        The process involves:
        1. Projecting the input into Query, Key, and Value spaces.
        2. Splitting the projections into multiple heads.
        3. Computing scaled dot-product attention weights.
        4. Applying optional architectural biases (e.g., from the pair representation).
        5. Applying an optional attention mask (e.g., for padding or causal masking).
        6. Aggregating the values and projecting back to the original input dimension.
        7. Optionally applying a sigmoid gate.

        Args:
            input_tensor: Input tensor of shape (..., number_residues, input_dimension).
            bias_tensor: Optional bias tensor of shape (..., number_heads, number_residues, number_residues)
                or broadcastable to this shape. Often derived from the pair representation.
            attention_mask_tensor: Optional binary mask tensor of shape (..., number_residues)
                where 1 indicates an active residue and 0 indicates a masked one.

        Returns:
            Output tensor of shape (..., number_residues, input_dimension).
        """
        query_embedding = self.query_embedder(input_tensor)
        key_embedding = self.key_embedder(input_tensor)
        value_embedding = self.value_embedder(input_tensor)

        query_embedding, key_embedding, value_embedding = self.separate_key_query_value_heads(
            query_embedding=query_embedding, key_embedding=key_embedding, value_embedding=value_embedding)

        # Compute Attention Tensor And Scale It Accordingly
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

            # Hard offset equivalent to minus infinity turning softmax to effectively 0 for these values.
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


if __name__ == "__main__":
    from pathlib import Path
    from utilities.tensor_utilities import get_device, print_tensor_shape
    from feature_extraction.extractor import FeatureExtractor
    from embedders.input_embedder import InputEmbedder

    # Robust path to the test file
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    msa_file_path = project_root / "tests" / "feature_extraction" / "multiple_sequence_alignement.a3m"

    # Initialize the extractor with fixed parameters and seed for determinism
    extractor = FeatureExtractor(
        file_path=str(msa_file_path),
        maximum_cluster_sequences=512,
        maximum_extra_msa_sequences=5120,
        mask_probability=0.15,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=0
    )

    computer_device = get_device()

    if str(computer_device) == "mps":
        tensor_dtype = torch.float32
    else:
        tensor_dtype = torch.float64

    # Initialize Input Embedder
    input_embedder = InputEmbedder(
        input_sequence_feature_dimension=extractor.input_sequence_feature.shape[-1],
        input_msa_feature_dimension=extractor.input_msa_feature.shape[-1],
        input_extra_msa_feature_dimension=extractor.input_extra_msa_feature.shape[-1],
        msa_embedding=256,
        extra_msa_embedding=64,
        pair_representation_embedding=10,
        number_neighbouring_amino_acids=32,
        device=computer_device,
        dtype=tensor_dtype
    )

    msa_rep, pair_rep, extra_msa_rep = input_embedder(
        input_sequence_feature=extractor.input_sequence_feature.to(device=computer_device, dtype=tensor_dtype),
        input_msa_feature=extractor.input_msa_feature.to(device=computer_device, dtype=tensor_dtype),
        input_residue_index_feature=extractor.input_residue_index_feature.to(device=computer_device,
                                                                             dtype=tensor_dtype),
        input_extra_msa_feature=extractor.input_extra_msa_feature.to(device=computer_device, dtype=tensor_dtype),
    )

    multi_head_attention = MultiHeadAttention(
        input_dimension=msa_rep.shape[-1],
        head_embedding_dimension=48,
        number_heads=10,
        attention_dimension=-2,
        use_gating=True,
        use_global_attention=False,
        use_embedding_bias=True,
        device=computer_device,
        dtype=tensor_dtype)

    output_multi_head_attention_tensor = multi_head_attention(input_tensor=msa_rep,
                                                              bias_tensor=pair_rep.movedim(source=-1, destination=-3),
                                                              attention_mask_tensor=None)

    print_tensor_shape(name="MSA Representation", tensor=msa_rep)
    print_tensor_shape(name="Pair Representation", tensor=pair_rep)
    print_tensor_shape(name="Extra MSA Representation", tensor=extra_msa_rep)
    print_tensor_shape(name="Output MHA Of MSA Representation", tensor=output_multi_head_attention_tensor)
