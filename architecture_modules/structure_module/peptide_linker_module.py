import torch
from torch import nn


class PeptideLinkerResNetLayer(nn.Module):
    """
    A residual layer used within the PeptideLinkerPredictor.
    """

    def __init__(self, peptide_linker_representation_embedding: int, device: torch.device, dtype: torch.dtype):
        """
        Initializes the PeptideLinkerResNetLayer.

        Args:
            peptide_linker_representation_embedding (int): Hidden dimension for scaler prediction.
            device (torch.device): Target device.
            dtype (torch.dtype): Target data type.
        """
        super().__init__()
        self.peptide_linker_representation_embedding = peptide_linker_representation_embedding
        self.device = device
        self.dtype = dtype

        self.relu = nn.ReLU()
        self.first_embedder = nn.Linear(in_features=self.peptide_linker_representation_embedding,
                                        out_features=self.peptide_linker_representation_embedding,
                                        device=self.device, dtype=self.dtype)
        self.second_embedder = nn.Linear(in_features=self.peptide_linker_representation_embedding,
                                         out_features=self.peptide_linker_representation_embedding,
                                         device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a residual block: x + Linear(ReLU(Linear(ReLU(x))))
        """
        residual = x
        x = self.relu(x)
        x = self.first_embedder(x)
        x = self.relu(x)
        x = self.second_embedder(x)
        return x + residual


class PeptideLinkerPredictor(nn.Module):
    """
    Predicts the 4 invariant scalers for the peptide linker between residue i and i+1.
    """

    def __init__(self, single_representation_embedding: int, peptide_linker_representation_embedding: int,
                 device: torch.device, dtype: torch.dtype):
        """
        Initializes the PeptideLinkerPredictor.

        Args:
            single_representation_embedding (int): Feature dimension of the single representation.
            peptide_linker_representation_embedding (int): Hidden dimension for scaler prediction.
            device (torch.device): Target device.
            dtype (torch.dtype): Target data type.
        """
        super().__init__()
        self.single_representation_embedding = single_representation_embedding
        self.peptide_linker_representation_embedding = peptide_linker_representation_embedding
        self.device = device
        self.dtype = dtype

        self.initial_single_representation_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                                                out_features=self.peptide_linker_representation_embedding,
                                                                device=self.device, dtype=self.dtype)
        self.current_single_representation_embedder = nn.Linear(in_features=self.single_representation_embedding,
                                                                out_features=self.peptide_linker_representation_embedding,
                                                                device=self.device, dtype=self.dtype)

        self.resnet_layers = nn.ModuleList([
            PeptideLinkerResNetLayer(peptide_linker_representation_embedding=self.peptide_linker_representation_embedding,
                                     device=self.device, dtype=self.dtype),
            PeptideLinkerResNetLayer(peptide_linker_representation_embedding=self.peptide_linker_representation_embedding,
                                     device=self.device, dtype=self.dtype)
        ])

        # We predict 4 scalers: Elevation, Angle(CA-C-N), Bond(C-N), Angle(C-N-CA)
        self.output_embedder = nn.Linear(in_features=self.peptide_linker_representation_embedding,
                                         out_features=4,
                                         device=self.device, dtype=self.dtype)
        self.relu = nn.ReLU()

    def forward(self, single_representation: torch.Tensor, initial_single_representation: torch.Tensor) -> torch.Tensor:
        """
        Predicts the 4 peptide linker scalers.

        Args:
            single_representation (torch.Tensor): Current single representation. Shape `(..., number_residues, single_rep_dim)`
            initial_single_representation (torch.Tensor): Initial single representation. Shape `(..., number_residues, single_rep_dim)`

        Returns:
            torch.Tensor: Predicted peptide linker scalers. Shape `(..., number_residues, 4)`
        """
        initial_embedding = self.initial_single_representation_embedder(initial_single_representation)
        current_embedding = self.current_single_representation_embedder(single_representation)

        x = self.relu(initial_embedding + current_embedding)

        for layer in self.resnet_layers:
            x = layer(x)

        raw_outputs = self.output_embedder(x)

        # Apply sigmoid to strictly bound the outputs
        # Shape: (..., number_residues, 4)
        base_scalers = torch.sigmoid(raw_outputs) + 0.5
        
        # Adjust the elevation scaler (index 0) to be zero-centered between -10.0 and 10.0
        # Create a clone to avoid in-place modification issues with autograd gradients in some pytorch versions
        scalers = base_scalers.clone()
        scalers[..., 0] = (torch.sigmoid(raw_outputs[..., 0]) - 0.5) * 20.0

        return scalers
