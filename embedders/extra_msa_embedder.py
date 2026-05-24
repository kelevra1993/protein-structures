import torch
from torch import nn

from architecture_modules.attention_module.multi_head_attention import MultiHeadAttention
from architecture_modules.evoformer_module.msa_stack import MSARowAttentionWithPairBias, MSATransition, OuterProductMean
from architecture_modules.evoformer_module.pair_stack import PairStack, PairTransition


class ExtraMsaEmbedder(nn.Module):
    """
    The `ExtraMsaEmbedder` linearly projects the input extra MSA features into the 
    extra MSA embedding dimension. This representation is subsequently used by the 
    Extra MSA Stack.
    """

    def __init__(self, input_extra_msa_feature_dimension: int, extra_msa_embedding: int, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Initializes the ExtraMsaEmbedder module.

        Args:
            input_extra_msa_feature_dimension (int): The dimension of the input extra MSA features.
            extra_msa_embedding (int): The hidden dimension for the extra MSA representation.
            device (torch.device): The device on which to initialize the tensors.
            dtype (torch.dtype): The data type for the tensors.
        """
        super().__init__()
        self.input_extra_msa_feature_dimension = input_extra_msa_feature_dimension
        self.extra_msa_embedding = extra_msa_embedding
        self.device = device
        self.dtype = dtype

        self.linear = nn.Linear(in_features=self.input_extra_msa_feature_dimension,
                                out_features=self.extra_msa_embedding, device=self.device, dtype=self.dtype)

    def forward(self, input_extra_msa_feature: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input extra MSA features into the extra MSA representation.

        Args:
            input_extra_msa_feature (torch.Tensor): The raw extra MSA features.
                Shape: (*, number_extra_sequences, number_residues, input_extra_msa_feature_dimension)

        Returns:
            torch.Tensor: The embedded extra MSA representation.
                Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)
        """
        extra_msa_representation = self.linear(input_extra_msa_feature)
        return extra_msa_representation


class MSAColumnGlobalAttention(nn.Module):
    """
    Implements the MSA Column Global Attention mechanism (Algorithm 19).
    This module performs global attention over the columns (sequences) of the Extra MSA representation.
    """

    def __init__(self, msa_embedding: int, head_embedding_dimension: int, number_heads: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the MSAColumnGlobalAttention module.

        Args:
            msa_embedding (int): Hidden dimension size for the MSA representation.
            device (torch.device): The device on which to initialize the tensors.
            dtype (torch.dtype): The data type for the tensors.
            head_embedding_dimension (int): Embedding dimension per attention head.
            number_heads (int, optional): Number of attention heads.
        """
        super().__init__()
        self.msa_embedding = msa_embedding
        self.head_embedding_dimension = head_embedding_dimension
        self.number_heads = number_heads
        self.device = device
        self.dtype = dtype

        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.msa_embedding,
                                                                device=self.device,
                                                                dtype=self.dtype)

        # The use of global attention is the first thing that differs from the classic evormer that takes in the
        # msa_representation and the pair_representation.
        self.global_multi_head_attention = MultiHeadAttention(input_dimension=self.msa_embedding,
                                                              head_embedding_dimension=self.head_embedding_dimension,
                                                              number_heads=self.number_heads,
                                                              attention_dimension=-3,
                                                              use_gating=True,
                                                              use_global_attention=True,
                                                              use_embedding_bias=False,
                                                              device=self.device,
                                                              dtype=self.dtype)

    def forward(self, msa_representation: torch.Tensor) -> torch.Tensor:
        """
        Applies global multi-head attention across the sequence dimension (columns).

        Args:
            msa_representation (torch.Tensor): The input Extra MSA representation.
                Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)

        Returns:
            torch.Tensor: The globally attended Extra MSA representation.
                Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)
        """
        normalized_msa_representation = self.msa_representation_layer_normalizer(msa_representation)
        output_tensor = self.global_multi_head_attention.forward(input_tensor=normalized_msa_representation)

        return output_tensor


class ExtraMsaBlock(nn.Module):
    """
    Implements one block for the Extra MSA Stack (Algorithm 18).
    Updates the Extra MSA representation and the Pair representation.
    """

    def __init__(self, extra_msa_embedding: int, pair_representation_embedding: int, msa_global_attention_heads: int,
                 msa_global_attention_head_embeddings: int, number_heads: int, head_embedding_dimension: int,
                 intermediate_embedding: int, channel_scaler: int, triangle_multiplication_embedding: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the ExtraMsaBlock module.

        Args:
            extra_msa_embedding (int): Hidden dimension size for the Extra MSA representation.
            pair_representation_embedding (int): Hidden dimension size for the pair representation.
            device (torch.device): The device on which to initialize the tensors.
            dtype (torch.dtype): The data type for the tensors.
            msa_global_attention_heads (int): Number of attention heads for global attention.
            msa_global_attention_head_embeddings (int): Embedding dimension per attention head for global attention.
            number_heads (int): Number of attention heads.
            head_embedding_dimension (int): Embedding dimension per attention head.
            intermediate_embedding (int): Intermediate dimension for OuterProductMean.
            channel_scaler (int): Channel scaler for transition layers.
            triangle_multiplication_embedding (int): Dimension for triangle multiplication.
        """
        super().__init__()
        self.extra_msa_embedding = extra_msa_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.device = device
        self.dtype = dtype

        self.extra_msa_row_wise_attention = MSARowAttentionWithPairBias(
            msa_embedding=self.extra_msa_embedding,
            pair_representation_embedding=self.pair_representation_embedding,
            head_embedding_dimension=head_embedding_dimension,
            number_heads=number_heads,
            device=self.device,
            dtype=self.dtype)

        # Here we want the control of the MSAColumnGlobalAttention that is why we have it in our arguments.
        self.extra_msa_global_column_wise_attention = MSAColumnGlobalAttention(
            msa_embedding=self.extra_msa_embedding,
            head_embedding_dimension=msa_global_attention_head_embeddings,
            number_heads=msa_global_attention_heads,
            device=self.device,
            dtype=self.dtype)

        self.extra_msa_transition_embedder = MSATransition(
            msa_embedding=self.extra_msa_embedding,
            channel_scaler=channel_scaler,
            device=self.device,
            dtype=self.dtype)

        self.outer_product_mean = OuterProductMean(
            msa_embedding=self.extra_msa_embedding,
            pair_representation_embedding=self.pair_representation_embedding,
            intermediate_embedding=intermediate_embedding,
            device=self.device,
            dtype=self.dtype)

        self.pair_stack_embedder = PairStack(
            pair_representation_dimension=self.pair_representation_embedding,
            head_embedding_dimension=head_embedding_dimension,
            triangle_multiplication_embedding=triangle_multiplication_embedding,
            number_heads=number_heads,
            channel_scaler=channel_scaler,
            device=self.device,
            dtype=self.dtype)

    def forward(self, extra_msa_representation: torch.Tensor,
                pair_representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the Extra MSA and pair representations through the block.

        Args:
            extra_msa_representation (torch.Tensor): The Extra MSA representation.
                Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)
            pair_representation (torch.Tensor): The pair representation.
                Shape: (*, number_residues, number_residues, pair_representation_dimension)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - extra_msa_representation (torch.Tensor): Updated Extra MSA representation.
                    Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)
                - pair_representation (torch.Tensor): Updated pair representation.
                    Shape: (*, number_residues, number_residues, pair_representation_dimension)
        """
        extra_msa_representation = extra_msa_representation + self.extra_msa_row_wise_attention(
            msa_representation=extra_msa_representation, pair_representation=pair_representation)

        extra_msa_representation = extra_msa_representation + self.extra_msa_global_column_wise_attention(
            msa_representation=extra_msa_representation)

        extra_msa_representation = extra_msa_representation + self.extra_msa_transition_embedder(
            msa_representation=extra_msa_representation)

        pair_representation = pair_representation + self.outer_product_mean(msa_representation=extra_msa_representation)
        pair_representation = self.pair_stack_embedder(pair_representation=pair_representation)

        return extra_msa_representation, pair_representation


class ExtraMsaStack(nn.Module):
    """
    Implements the Extra MSA Stack (Algorithm 18).
    Applies a series of ExtraMsaBlocks to iteratively update the extra MSA representation
    and the pair representation.
    """

    def __init__(self, extra_msa_embedding: int, pair_representation_embedding: int, number_blocks: int,
                 msa_global_attention_heads: int, msa_global_attention_head_embeddings: int, number_heads: int,
                 head_embedding_dimension: int, intermediate_embedding: int, channel_scaler: int,
                 triangle_multiplication_embedding: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the ExtraMsaStack module.

        Args:
            extra_msa_embedding (int): Hidden dimension size for the Extra MSA representation.
            pair_representation_embedding (int): Hidden dimension size for the pair representation.
            number_blocks (int): Number of Extra MSA blocks to instantiate.
            device (torch.device): The device on which to initialize the tensors.
            dtype (torch.dtype): The data type for the tensors.
            msa_global_attention_heads (int): Number of attention heads for global attention.
            msa_global_attention_head_embeddings (int): Embedding dimension per attention head for global attention.
            number_heads (int): Number of attention heads.
            head_embedding_dimension (int): Embedding dimension per attention head.
            intermediate_embedding (int): Intermediate dimension for OuterProductMean.
            channel_scaler (int): Channel scaler for transition layers.
            triangle_multiplication_embedding (int): Dimension for triangle multiplication.
        """
        super().__init__()
        self.extra_msa_embedding = extra_msa_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.number_blocks = number_blocks
        self.device = device
        self.dtype = dtype

        self.extra_msa_evoformer_blocks = nn.ModuleList([
            ExtraMsaBlock(
                extra_msa_embedding=self.extra_msa_embedding,
                pair_representation_embedding=self.pair_representation_embedding,
                device=self.device,
                dtype=self.dtype,
                msa_global_attention_heads=msa_global_attention_heads,
                msa_global_attention_head_embeddings=msa_global_attention_head_embeddings,
                number_heads=number_heads,
                head_embedding_dimension=head_embedding_dimension,
                intermediate_embedding=intermediate_embedding,
                channel_scaler=channel_scaler,
                triangle_multiplication_embedding=triangle_multiplication_embedding
            ) for _ in range(self.number_blocks)
        ])

    def forward(self, extra_msa_representation: torch.Tensor, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Processes the representations through the sequence of Extra MSA blocks.

        Args:
            extra_msa_representation (torch.Tensor): The Extra MSA representation.
                Shape: (*, number_extra_sequences, number_residues, extra_msa_embedding_dimension)
            pair_representation (torch.Tensor): The pair representation.
                Shape: (*, number_residues, number_residues, pair_representation_dimension)

        Returns:
            torch.Tensor: The updated pair representation.
                Shape: (*, number_residues, number_residues, pair_representation_dimension)
        """
        for block in self.extra_msa_evoformer_blocks:
            extra_msa_representation, pair_representation = block(
                extra_msa_representation=extra_msa_representation,
                pair_representation=pair_representation
            )

        return pair_representation
