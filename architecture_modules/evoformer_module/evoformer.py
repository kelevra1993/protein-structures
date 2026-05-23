from torch import nn

from architecture_modules.evoformer_module.msa_stack import (MSARowAttentionWithPairBias, MSAColumnAttention,
                                                             MSATransition, OuterProductMean)
from architecture_modules.evoformer_module.pair_stack import PairStack


class EvoformerBlock(nn.Module):

    def __init__(self, msa_embedding, pair_representation_embedding, device, dtype):
        super().__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(msa_embedding=msa_embedding,
                                                       pair_representation_embedding=pair_representation_embedding)
        self.msa_att_col = MSAColumnAttention(msa_embedding=msa_embedding)
        self.msa_transition = MSATransition(msa_embedding=msa_embedding)
        self.outer_product_mean = OuterProductMean(msa_embedding=msa_embedding,
                                                   pair_representation_embedding=pair_representation_embedding)

        self.core = PairStack(pair_representation_embedding)

    def forward(self, msa_representation, pair_representation):
        msa_representation = msa_representation + self.msa_att_row(msa_representation=msa_representation,
                                                                   pair_representation=pair_representation)
        msa_representation += self.msa_att_col(msa_representation)
        msa_representation += self.msa_transition(msa_representation)

        pair_representation = pair_representation + self.outer_product_mean(msa_representation)
        pair_representation = self.core(pair_representation)

        return msa_representation, pair_representation


class EvoformerStack(nn.Module):

    def __init__(self, msa_embedding, pair_representation_embedding, number_blocks, single_representation_embedding,
                 device, dtype):
        super().__init__()

        self.blocks = nn.ModuleList(
            [EvoformerBlock(msa_embedding=msa_embedding, pair_representation_embedding=pair_representation_embedding)
             for i in range(number_blocks)])

        self.linear = nn.Linear(in_features=msa_embedding, out_features=single_representation_embedding)

    def forward(self, msa_representation, pair_representation):
        for block in self.blocks:
            msa_representation, pair_representation = block(msa_representation, pair_representation)

        single_representation = self.linear(msa_representation[..., 0, :, :])

        return msa_representation, pair_representation, single_representation
