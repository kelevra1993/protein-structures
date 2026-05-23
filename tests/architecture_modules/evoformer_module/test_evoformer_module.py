import math
import torch

batch_size = 3
msa_embedding = 4
pair_representation_embedding = 5
embedding_dimension = 6
number_heads = 7
number_sequences = 8
number_residues = 9

msa_representation_shape = (number_sequences, number_residues, msa_embedding)
pair_representation_shape = (number_residues, number_residues, pair_representation_embedding)
msa_representation_shape_batched = (batch_size,) + msa_representation_shape
pair_representation_shape_batched = (batch_size,) + pair_representation_shape

# MSA Representation Testing Input
msa_representation = torch.linspace(
    start=-2, end=2, steps=math.prod(msa_representation_shape)).reshape(msa_representation_shape)
msa_representation_batch = torch.linspace(
    start=-2, end=2, steps=math.prod(msa_representation_shape_batched)).reshape(msa_representation_shape_batched)

# Pair Representation Testing Input
pair_representation = torch.linspace(
    start=-2, end=2, steps=math.prod(pair_representation_shape)).reshape(pair_representation_shape)
pair_representation_batch = torch.linspace(
    start=-2, end=2, steps=math.prod(pair_representation_shape_batched)).reshape(pair_representation_shape_batched)

inputs = {
    'msa_representation': (msa_representation.double(), msa_representation_batch.double()),
    'pair_representation': (pair_representation.double(), pair_representation_batch.double()),
}
