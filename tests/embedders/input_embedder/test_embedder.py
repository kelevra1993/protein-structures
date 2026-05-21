import math
import torch
from typing import List

batch_size = 3
msa_embedding_dimension = 4
pair_representation_dimension = 5
general_embedding_dimension = 6
number_heads = 7
number_clusters = 8
number_extra_sequences = 9
number_residues = 35
msa_feature_dimension = 10
input_sequence_feature_dimension = 11
input_extra_msa_feature_dimension = 12
extra_msa_embedding_dimension = 13

feature_shapes = {
    'msa_feature': (number_clusters, number_residues, msa_feature_dimension),
    'input_sequence_feature': (number_residues, input_sequence_feature_dimension),
    'residue_index': (number_residues,),
    'input_extra_msa_feature': (number_extra_sequences, number_residues, input_extra_msa_feature_dimension),
}

output_shapes = {
    'msa_representation': (number_clusters, number_residues, msa_embedding_dimension),  # Outputs not needed
    'pair_representation': (number_residues, number_residues, pair_representation_dimension),  # Outputs not needed
    'extra_msa_representation': (number_extra_sequences, number_residues, extra_msa_embedding_dimension),
    # Outputs not needed
}

batched_feature_shapes = {
    key: (batch_size,) + value
    for key, value in feature_shapes.items()
}

test_inputs = {
    key: torch.linspace(-2, 2, math.prod(shape)).reshape(shape).double()
    for key, shape in feature_shapes.items()
}

test_inputs['residue_index'] = torch.arange(number_residues)
