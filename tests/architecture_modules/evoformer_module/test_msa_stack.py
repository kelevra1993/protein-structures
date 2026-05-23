import torch
from tests.utilities.testing_utilities import get_evoformer_test_inputs

config, test_inputs = get_evoformer_test_inputs()

# Unpack configuration variables
batch_size = config['batch_size']
msa_embedding = config['msa_embedding']
pair_representation_embedding = config['pair_representation_embedding']
embedding_dimension = config['embedding_dimension']
number_heads = config['number_heads']
number_sequences = config['number_sequences']
number_residues = config['number_residues']
