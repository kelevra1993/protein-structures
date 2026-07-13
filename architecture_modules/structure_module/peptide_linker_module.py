import torch
from torch import nn


class PeptideLinkerResNetLayer(nn.Module):
    """
    A standard residual layer block used within the PeptideLinkerPredictor.

    This layer acts as a feature refinement block, applying a two-layer multi-layer perceptron 
    with ReLU activations. A residual connection is added from the input tensor to the output 
    of the second linear layer to facilitate gradient flow and stable deep feature extraction.
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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the residual block.

        Args:
            input_tensor (torch.Tensor): The input hidden representation tensor.
                Shape: `(..., number_residues, peptide_linker_representation_embedding)`

        Returns:
            torch.Tensor: The refined hidden representation tensor after applying
                the two-layer MLP and adding the original input_tensor (residual connection).
                Shape: `(..., number_residues, peptide_linker_representation_embedding)`
        """
        residual_tensor = input_tensor
        output_tensor = self.second_embedder(self.relu(self.first_embedder(self.relu(input_tensor))))

        return output_tensor + residual_tensor


class PeptideLinkerPredictor(nn.Module):
    """
    Predicts the 4 invariant geometric scalers for the peptide linker connecting adjacent residues.

    The PeptideLinkerPredictor replaces the standard Backbone Update mechanism by formulating 
    the construction of consecutive amino acids iteratively via relative 3D geometric transformations 
    rather than absolute rigid body updates. To maintain physical plausibility, it predicts 
    4 structural scalars required for reconstructing the peptide backbone from residue `i` to `i+1`:
    1. Elevation Angle Scaler (for determining the position of the next Nitrogen).
    2. Angle Scaler (CA-C-N).
    3. Bond Length Scaler (C-N).
    4. Angle Scaler (C-N-CA).

    These scalers are derived by combining the initial structural single representation (providing 
    the global sequence context) with the current refined single representation (providing the 
    current structural hypotheses), passing the combined representation through a ResNet tower, 
    and outputting strictly bounded scaling factors.
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

        self.initial_single_representation_embedder = nn.Linear(
            in_features=self.single_representation_embedding,
            out_features=self.peptide_linker_representation_embedding,
            device=self.device, dtype=self.dtype)
        self.current_single_representation_embedder = nn.Linear(
            in_features=self.single_representation_embedding,
            out_features=self.peptide_linker_representation_embedding,
            device=self.device, dtype=self.dtype)

        self.layer_normalizer = nn.LayerNorm(self.peptide_linker_representation_embedding,
                                             device=self.device, dtype=self.dtype)

        self.resnet_layers = nn.ModuleList([
            PeptideLinkerResNetLayer(
                peptide_linker_representation_embedding=self.peptide_linker_representation_embedding,
                device=self.device, dtype=self.dtype),
            PeptideLinkerResNetLayer(
                peptide_linker_representation_embedding=self.peptide_linker_representation_embedding,
                device=self.device, dtype=self.dtype)
        ])

        # We predict 4 scalers: Elevation, Angle(CA-C-N), Bond(C-N), Angle(C-N-CA)
        self.output_embedder = nn.Linear(in_features=self.peptide_linker_representation_embedding,
                                         out_features=4,
                                         device=self.device, dtype=self.dtype)
        self.relu = nn.ReLU()

    def forward(self, single_representation: torch.Tensor, initial_single_representation: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass to predict the 4 structural peptide linker scalers.

        The initial and current single representations are embedded to a common hidden 
        dimension, summed, and passed through a series of residual blocks. The final 
        output is generated via an embedder and bounded using a sigmoid function. 
        Bond length and standard angle scalers are centered around 1.0 (to act as 
        multipliers for ideal physical geometries), while the elevation scaler is 
        centered around 0 and bounded between -10.0 and 10.0.

        Args:
            single_representation (torch.Tensor): The structurally refined single representation.
                Shape: `(..., number_residues, single_representation_embedding)`
            initial_single_representation (torch.Tensor): The original input single representation 
                before any structural module refinement.
                Shape: `(..., number_residues, single_representation_embedding)`

        Returns:
            torch.Tensor: The predicted geometric scalers for the peptide linker.
                The last dimension contains: [elevation, ca_c_n_angle, c_n_length, c_n_ca_angle].
                Shape: `(..., number_residues, 4)`
        """
        initial_embedded_representation = self.initial_single_representation_embedder(initial_single_representation)
        current_embedded_representation = self.current_single_representation_embedder(single_representation)

        initial_embedded_representation = self.layer_normalizer(initial_embedded_representation)
        current_embedded_representation = self.layer_normalizer(current_embedded_representation)

        hidden_representation = self.relu(initial_embedded_representation + current_embedded_representation)

        for layer in self.resnet_layers:
            hidden_representation = layer(hidden_representation)
            hidden_representation = self.layer_normalizer(hidden_representation)

        raw_outputs = self.output_embedder(hidden_representation)

        # Apply sigmoid to strictly bound the outputs between 0.5 and 1.5
        # Shape: (..., number_residues, 4)
        base_scalers = torch.sigmoid(raw_outputs) + 0.5

        # Adjust the elevation scaler (index 0) to be zero-centered between -10.0 and 10.0
        # Create a clone to avoid in-place modification issues with autograd gradients in some pytorch versions
        scalers = base_scalers.clone()
        scalers[..., 0] = (torch.sigmoid(raw_outputs[..., 0]) - 0.5) * 20.0

        return scalers
