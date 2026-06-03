from typing import Tuple

import torch
from torch import nn


class LddtModule(nn.Module):
    """
    Implements the lDDT (Local Distance Difference Test) Module, a confidence head in AlphaFold II.

    The lDDT Module (often referred to as the pLDDT head) predicts the local structure 
    quality of the protein model on a per-residue basis. It takes the final 
    single representation as input and projects it into 50 discrete bins representing 
    different lDDT score ranges. This provides a measure of the model's confidence 
    in its own structural prediction.

    The 50 bins typically represent lDDT scores from 0 to 100 in increments of 2.
    """

    def __init__(self, single_representation_embedding: int, intermediate_embedding: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the LddtModule.

        Args:
            single_representation_embedding: The feature dimension of the input single representation.
            intermediate_embedding: The hidden dimension for the intermediate transition layers.
            device: The torch device on which the module's parameters will be allocated.
            dtype: The torch data type for the module's parameters.
        """
        super().__init__()

        self.single_representation_embedding = single_representation_embedding
        self.intermediate_embedding = intermediate_embedding
        self.device = device
        self.dtype = dtype

        self.single_representation_layer_normalizer = nn.LayerNorm(
            normalized_shape=self.single_representation_embedding,
            device=self.device, dtype=self.dtype)

        self.first_linear_layer = nn.Linear(in_features=self.single_representation_embedding,
                                            out_features=self.intermediate_embedding,
                                            device=self.device, dtype=self.dtype)

        self.second_linear_layer = nn.Linear(in_features=self.intermediate_embedding,
                                             out_features=self.intermediate_embedding,
                                             device=self.device, dtype=self.dtype)

        # Layer that projects to the predicted 50 bins ranging [1, 3, 5, ..., 99]
        self.third_linear_layer = nn.Linear(in_features=self.intermediate_embedding, out_features=50,
                                            device=self.device, dtype=self.dtype)

        self.relu = nn.ReLU()

    def forward(self, single_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass to predict lDDT confidence scores.

        The single representation is passed through a layer normalization and 
        a series of linear layers with ReLU activations to produce logits 
        for 50 lDDT bins.

        Args:
            single_representation: The input single representation tensor.
                Expected shape: (..., number_residues, single_representation_embedding)

        Returns:
            lddt_logits: The unnormalized predicted lDDT bin logits.
                Shape: (..., number_residues, 50)
            lddt_probabilities: Probability distribution over 50 lDDT bins.
                Shape: (..., number_residues, 50)
        """

        normalized_representation = self.single_representation_layer_normalizer(single_representation)

        # activation = relu(Linear(relu(Linear(normalized_representation))))
        activation = self.relu(self.second_linear_layer(self.relu(self.first_linear_layer(normalized_representation))))

        # Get Logits and Softmax
        lddt_logits = self.third_linear_layer(activation)
        lddt_probabilities = torch.softmax(lddt_logits, dim=-1)

        return lddt_logits, lddt_probabilities
