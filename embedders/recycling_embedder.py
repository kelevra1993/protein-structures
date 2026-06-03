import torch
from torch import nn
from utilities.tensor_utilities import specialised_one_hot_encoder


class RecyclingEmbedder(nn.Module):
    """
    The `RecyclingEmbedder` integrates structural and representation information from the
    previous recycling iteration of the AlphaFold II network. It takes the output MSA representation,
    pair representation, and predicted pseudo-Carbon-Beta (Cb) coordinates from the previous pass,
    and generates updates to initialize the representations for the current pass. This enables the
    network to iteratively refine the predicted protein structure.
    """

    def __init__(self, msa_embedding: int, pair_representation_embedding: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the RecyclingEmbedder module.

        Args:
            msa_embedding (int): Hidden dimension size for the MSA representation.
            pair_representation_embedding (int): Hidden dimension size for the pair representation.
            device (torch.device): The device on which to initialize the tensors (e.g., CPU or CUDA).
            dtype (torch.dtype): The data type for the tensors.
        """
        super().__init__()
        self.msa_embedding = msa_embedding
        self.pair_representation_embedding = pair_representation_embedding
        self.device = device
        self.dtype = dtype

        # TODO Note of importance for stop grad in the recycling embedder that will have to be implemented later.

        # TODO Note : This bin implementation might not be
        #  the one that we ultimately use so we have to be careful.
        self.bin_start = 3.25
        self.bin_end = 20.75
        self.bin_count = 15

        self.pair_distance_embedder = nn.Linear(in_features=self.bin_count,
                                                out_features=self.pair_representation_embedding,
                                                device=self.device, dtype=self.dtype)
        self.msa_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.msa_embedding,
                                                                device=self.device, dtype=self.dtype)
        self.pair_representation_layer_normalizer = nn.LayerNorm(normalized_shape=self.pair_representation_embedding,
                                                                 device=self.device, dtype=self.dtype)

        # Note : Openfold's implementation.
        # Alphafold's implementation uses closest distance to determine bin position
        # Registered as buffers to ensure they move with the module but are not trainable.
        self.register_buffer("bins", torch.linspace(start=self.bin_start, end=self.bin_end, steps=self.bin_count,
                                                    device=self.device, dtype=self.dtype))
        self.register_buffer("displaced_bins", torch.cat(tensors=(torch.linspace(start=self.bin_start,
                                                                                 end=self.bin_end,
                                                                                 steps=self.bin_count,
                                                                                 device=self.device,
                                                                                 dtype=self.dtype)[1:],
                                                                  torch.tensor([1e8], device=self.device,
                                                                               dtype=self.dtype)), dim=-1))

    def forward(self, previous_msa_representation: torch.Tensor,
                previous_pair_representation: torch.Tensor,
                previous_pseudo_carbon_beta_positions: torch.Tensor,
                use_alpha_fold_implementation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the recycling updates for the MSA and pair representations based on the previous iteration's outputs.

        This method calculates a distance matrix from the previous pseudo-Carbon-Beta positions,
        discretizes it into bins, and embeds it to update the pair representation. It also extracts
        and normalizes the query sequence (first cluster/row) from the previous MSA representation
        to update the new MSA representation.

        Args:
            previous_msa_representation (torch.Tensor): The MSA representation from the previous iteration.
                Shape: (*, number_clusters, number_residues, msa_embedding_dimension)
            previous_pair_representation (torch.Tensor): The pair representation from the previous iteration.
                Shape: (*, number_residues, number_residues, pair_representation_dimension)
            previous_pseudo_carbon_beta_positions (torch.Tensor):
            The predicted pseudo-Carbon-Beta coordinates from the previous structure module execution.
                Shape: (*, number_residues, 3)
            use_alpha_fold_implementation (bool, optional):
            If True, uses the strict AlphaFold II distance encoding (via specialised_one_hot_encoder).
             If False, uses the OpenFold continuous binning implementation. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - msa_representation (torch.Tensor):
                The recycled update for the MSA representation (query sequence only).
                    Shape: (*, number_residues, msa_embedding_dimension)
                - pair_representation (torch.Tensor): The recycled update for the pair representation.
                    Shape: (*, number_residues, number_residues, pair_representation_dimension)
        """
        difference_matrix = previous_pseudo_carbon_beta_positions.unsqueeze(
            -2) - previous_pseudo_carbon_beta_positions.unsqueeze(-3)
        distance_matrix = torch.linalg.vector_norm(difference_matrix, dim=-1)

        # Actual Alpha fold implementation vs Open fold's implementation
        if use_alpha_fold_implementation:
            distance_matrix = specialised_one_hot_encoder(input_tensor=distance_matrix, bin_tensor=self.bins).to(
                previous_pseudo_carbon_beta_positions.dtype)
        else:
            distance_matrix = distance_matrix.unsqueeze(-1)
            distance_matrix = ((distance_matrix > self.bins) * (distance_matrix < self.displaced_bins)).type(
                previous_pseudo_carbon_beta_positions.dtype)

        # Pair Representation Input for the evoformer and extra msa stack
        pair_representation = (self.pair_distance_embedder(distance_matrix) +
                               self.pair_representation_layer_normalizer(previous_pair_representation))

        # MSA Representation Input for the evoformer and extra msa stack
        msa_representation = self.msa_representation_layer_normalizer(previous_msa_representation[..., 0, :, :])

        return msa_representation, pair_representation
