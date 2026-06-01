import torch
from torch import nn


class DistogramModule(nn.Module):
    """
    Implements the Distogram Module, a diagnostic head in the AlphaFold II architecture.

    The Distogram Module predicts the probability distribution of distances between all pairs 
    of residues in a protein. It takes the final pair representation and projects it into 
    a set of distance bins. This module is primarily used during training to provide 
    a auxiliary loss (distogram loss) that helps supervise the structural information 
    captured by the pair representation.

    The output represents a probability distribution over 64 distance bins, typically 
    spanning a range from 2Å to 22Å.
    """

    def __init__(self, pair_representation_embedding: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the DistogramModule.

        Args:
            pair_representation_embedding: The feature dimension of the input pair representation.
            device: The torch device on which the module's parameters will be allocated.
            dtype: The torch data type for the module's parameters.
        """
        super().__init__()

        self.pair_representation_embedding = pair_representation_embedding
        self.device = device
        self.dtype = dtype

        # Linear projection to 64 distance bins
        self.distogram_embedder = nn.Linear(
            in_features=self.pair_representation_embedding,
            out_features=64,
            device=self.device,
            dtype=self.dtype
        )

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to produce the distogram.

        The pair representation is first symmetrized to ensure the predicted distances 
        are consistent (dist(i, j) == dist(j, i)). It is then projected to the 
        bin space and normalized using a softmax.

        Args:
            pair_representation: The 2D pair representation tensor.
                Expected shape: (..., number_residues, number_residues, pair_representation_dimension)

        Returns:
            distogram_probabilities: Probability distribution over 64 distance bins.
                Shape: (..., number_residues, number_residues, 64)
        """

        # Symmetrize the pair representation: z_ij = z_ij + z_ji
        # The residue dimensions are at indices -2 and -3.
        pair_representation = pair_representation + torch.transpose(pair_representation, dim0=-2, dim1=-3)

        # Project to logits
        logits = self.distogram_embedder(pair_representation)

        # Apply softmax to get probability distribution over the last dimension (bins)
        distogram_probabilities = torch.softmax(logits, dim=-1)

        return distogram_probabilities
