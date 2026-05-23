import math
import torch
from pathlib import Path

from embedders.input_embedder.embedder import InputEmbedder
from tests.utilities.testing_utilities import test_nn_module_method

batch_size = 3
msa_embedding_dimension = 4
pair_representation_dimension = 5
general_embedding_dimension = 6
number_clusters = 8
number_extra_sequences = 9
number_residues = 35
msa_feature_dimension = 10
input_sequence_feature_dimension = 11
input_extra_msa_feature_dimension = 12
extra_msa_embedding_dimension = 13

feature_shapes = {
    'input_msa_feature': (number_clusters, number_residues, msa_feature_dimension),
    'input_sequence_feature': (number_residues, input_sequence_feature_dimension),
    'input_residue_index_feature': (number_residues,),
    'input_extra_msa_feature': (number_extra_sequences, number_residues, input_extra_msa_feature_dimension),
}

output_tensor_names = [
    'msa_representation',
    'pair_representation',
    'extra_msa_representation',
]

test_input_tensors = {
    key: torch.linspace(-2, 2, math.prod(shape)).reshape(shape).double()
    for key, shape in feature_shapes.items()
}

test_input_tensors['input_residue_index_feature'] = torch.arange(number_residues)

# Initialize the module with dummy dimensions
input_embedder = InputEmbedder(
    input_sequence_feature_dimension=input_sequence_feature_dimension,
    input_msa_feature_dimension=msa_feature_dimension,
    input_extra_msa_feature_dimension=input_extra_msa_feature_dimension,
    msa_embedding=msa_embedding_dimension,
    extra_msa_embedding=extra_msa_embedding_dimension,
    pair_representation_embedding=pair_representation_dimension,
    number_neighbouring_amino_acids=32,
    device=torch.device("cpu"),
    dtype=torch.float64
)

# Run the test check
test_nn_module_method(
    module=input_embedder,
    input_tensor_dictionary=test_input_tensors,
    output_tensor_names=output_tensor_names,
    reference_folder=Path(__file__).parent / "reference_values",
    batch_size=batch_size
)

print(" - InputEmbedder Test Completed Successfuly.")
