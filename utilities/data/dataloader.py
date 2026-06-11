import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utilities.data.input import ModelInput
from utilities.os_utilities import read_json


class ProteinDataset(Dataset):
    """
    A PyTorch Dataset for loading protein structures and their associated MSAs.

    This dataset serves as the primary data source for training and validating the
    AlphaFold II model. It orchestrates the loading of structural data (.npz),
    sequence metadata (.json), and multiple sequence alignments (.a3m). It supports
    stochastic residue cropping, MSA clustering/masking via `ModelInput`, and
    data balancing through length-based acceptance filtering.

    The dataset is initialized from a split file that defines which clusters or
    individual proteins belong to the current training phase (Train, Validation, or Test).
    """

    def __init__(self, data_folder: str | Path, split_file_path: str | Path,
                 acceptance_slope_start: int,
                 acceptance_slope_end: int,
                 residue_crop_size: int | None,
                 emphasize_beginning_crops: bool,
                 distribution_threshold: int,
                 maximum_cluster_sequences: int,
                 maximum_extra_msa_sequences: int,
                 mask_probability: float,
                 number_recycle_cycles: int,
                 use_single_representative: bool,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        Initializes the ProteinDataset with configuration for data loading and processing.

        :param data_folder: Path to the root data folder containing 'structures', 'records', and 'raw_msa'.
        :param split_file_path: Path to the JSON file defining the train/validation split (keys are clusters, values are lists of protein IDs).
        :param acceptance_slope_start: Threshold for input filtering probability calculation.
        :param acceptance_slope_end: Upper threshold for input filtering probability calculation.
        :param residue_crop_size: The number of residues to crop from each sequence during sampling.
        :param emphasize_beginning_crops: If True, applies a stochastic bias to sample crops closer to the N-terminus.
        :param distribution_threshold: Threshold to filter out sequences with highly biased amino acid distributions.
        :param maximum_cluster_sequences: Max number of sequences for the main MSA clusters.
        :param maximum_extra_msa_sequences: Max number of sequences for the extra MSA stack.
        :param mask_probability: Probability of masking residues in MSA clusters.
        :param number_recycle_cycles: Number of recycling iterations to simulate in the batch (used by ModelInput.get_data).
        :param use_single_representative: If True, each cluster contributes only one random representative per epoch.
        :param device: The target torch.device.
        :param dtype: The target torch.dtype.
        """
        self.data_folder = Path(data_folder)
        self.split_file_path = Path(split_file_path)
        self.device = device
        self.dtype = dtype

        self.acceptance_slope_start = acceptance_slope_start
        self.acceptance_slope_end = acceptance_slope_end
        self.residue_crop_size = residue_crop_size
        self.emphasize_beginning_crops = emphasize_beginning_crops
        self.distribution_threshold = distribution_threshold
        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences
        self.mask_probability = mask_probability
        self.number_recycle_cycles = number_recycle_cycles
        self.use_single_representative = use_single_representative

        # Parse the JSON split file
        # The structure is assumed to be: { "cluster_id": ["protein_id_1", "protein_id_2"], ... }
        cluster_mapping = read_json(path=str(self.split_file_path))

        # Flatten the clusters into a single list of protein IDs
        # If we set use_single_representative as True, we just take one random element.
        self.protein_ids = []
        for cluster, members in cluster_mapping.items():
            if self.use_single_representative:
                self.protein_ids.append(random.choice(members))
            else:
                self.protein_ids.extend(members)

    def __len__(self) -> int:
        """
        Returns the total number of items in this dataset split. If we set use_single_representative as True,
        this is the number of clusters. Otherwise, it is the total number of proteins.
        """
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Todo we actually haven't implemented a mechanism of skipping datapoints
          here might seem the best place to do this but we are dealing with an index that is passed that we do
          not have control of unless we call __getitem__ explicitly which might defeat the purpose of our dataloader.

        Retrieves the cropped and processed tensor data for a specific protein.

        It constructs the paths to the .npz, .json, and .a3m files based on the
        protein ID, initializes a ModelInput, and returns the processed tensor dictionary.

        :param index: The integer index of the protein in the flattened list.
        :return: A dictionary containing all input and ground truth tensors for the model.
        """
        protein_id = self.protein_ids[index]

        structure_path = self.data_folder / "structures" / f"{protein_id}.npz"
        record_path = self.data_folder / "records" / f"{protein_id}.json"
        msa_path = self.data_folder / "raw_msa" / f"{protein_id}.a3m"

        # Initialize the ModelInput
        model_input = ModelInput(
            structure_path=str(structure_path),
            msa_path=str(msa_path),
            record_path=str(record_path),
            acceptance_slope_start=self.acceptance_slope_start,
            acceptance_slope_end=self.acceptance_slope_end,
            residue_crop_size=self.residue_crop_size,
            emphasize_beginning_crops=self.emphasize_beginning_crops,
            distribution_threshold=self.distribution_threshold,
            maximum_cluster_sequences=self.maximum_cluster_sequences,
            maximum_extra_msa_sequences=self.maximum_extra_msa_sequences,
            mask_probability=self.mask_probability,
            device=self.device,
            dtype=self.dtype)

        # Get the cropped and recycled batch data
        # NOTE : if self.residue_crop_size is None then we just get the whole data
        batch_data = model_input.get_data(number_samples=self.number_recycle_cycles, seed=None, batch_mode=False)

        return batch_data


def protein_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for batching protein dictionaries with potentially 
    different sequence lengths and different numbers of MSA sequences.

    It pads all tensors along every dimension to the maximum size present 
    in the current batch.

    :param batch: A list of dictionaries, each containing tensors for a single protein.
    :return: A single dictionary where all tensors are stacked into batches.
    """
    if not batch:
        return {}

    # Fast path for batch size 1 to avoid padding overhead
    if len(batch) == 1:
        return {key: value.unsqueeze(0) for key, value in batch[0].items()}

    collated_batch = {}
    keys = batch[0].keys()

    for key in keys:

        # 1. Find the maximum size for each dimension across the batch
        # We assume all tensors for this key have the same number of dimensions.
        # Generally either number_residues, or number_msa_cluster or number_extra_msa_cluster
        max_shape = list(batch[0][key].shape)

        for item in batch[1:]:
            current_shape = item[key].shape
            for i, size in enumerate(current_shape):
                if size > max_shape[i]:
                    max_shape[i] = size

        # 2. Pad each tensor to max_shape
        padded_tensors = []
        for item in batch:
            tensor = item[key]
            current_shape = list(tensor.shape)

            if current_shape != max_shape:
                # Create a zero tensor of the max shape
                padded_tensor = torch.zeros(max_shape, dtype=tensor.dtype, device=tensor.device)

                # Copy original data into the padded tensor
                # We create a tuple of slices [0:s1, 0:s2, ...]
                slices = [slice(0, s) for s in current_shape]
                padded_tensor[tuple(slices)] = tensor
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)

        # 3. Stack into batch (resulting shape will be (Batch, *max_shape))
        collated_batch[key] = torch.stack(padded_tensors, dim=0)

    return collated_batch


def get_protein_dataloader(data_folder: str | Path,
                           split_file_path: str | Path,
                           residue_crop_size: int | None,
                           emphasize_beginning_crops: bool,
                           acceptance_slope_start: int,
                           acceptance_slope_end: int,
                           distribution_threshold: int,
                           maximum_cluster_sequences: int,
                           maximum_extra_msa_sequences: int,
                           mask_probability: float,
                           number_recycle_cycles: int,
                           use_single_representative: bool,
                           batch_size: int = 1,
                           num_workers: int = 0,
                           shuffle: bool = False,
                           device: torch.device = torch.device("cpu"),
                           dtype: torch.dtype = torch.float32) -> DataLoader:
    """
    Constructs a PyTorch DataLoader for protein data with custom collation.

    This function abstracts the instantiation of `ProteinDataset` and `DataLoader`,
    ensuring that the `protein_collate_fn` is correctly applied to handle
    variable-sized protein features.

    Args:
        data_folder (str | Path): Path to the root data folder.
        split_file_path (str | Path): Path to the dataset split JSON file.
        residue_crop_size (int | None): Number of residues per crop.
        emphasize_beginning_crops (bool): Whether to bias crops towards the N-terminus.
        acceptance_slope_start (int): Start of the length-based acceptance ramp.
        acceptance_slope_end (int): End of the length-based acceptance ramp.
        distribution_threshold (int): Amino acid distribution filter threshold.
        maximum_cluster_sequences (int): Max sequences for MSA clusters.
        maximum_extra_msa_sequences (int): Max sequences for the extra MSA stack.
        mask_probability (float): MSA masking probability.
        number_recycle_cycles (int): Number of recycling iterations.
        use_single_representative (bool): If True, samples one protein per cluster.
        batch_size (int): Samples per batch.
        num_workers (int): Parallel worker threads.
        shuffle (bool): Whether to shuffle the dataset each epoch.
        device (torch.device): The target device.
        dtype (torch.dtype): The target data type.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.
    """

    dataset = ProteinDataset(
        data_folder=data_folder,
        split_file_path=split_file_path,
        acceptance_slope_start=acceptance_slope_start,
        acceptance_slope_end=acceptance_slope_end,
        residue_crop_size=residue_crop_size,
        emphasize_beginning_crops=emphasize_beginning_crops,
        distribution_threshold=distribution_threshold,
        maximum_cluster_sequences=maximum_cluster_sequences,
        maximum_extra_msa_sequences=maximum_extra_msa_sequences,
        mask_probability=mask_probability,
        number_recycle_cycles=number_recycle_cycles,
        use_single_representative=use_single_representative,
        device=device,
        dtype=dtype)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=protein_collate_fn,
        drop_last=False)

    return dataloader


def get_dataloader(data_folder: str,
                   model_configuration: dict,
                   split_path: str,
                   phase: str,
                   device: torch.device,
                   dtype: torch.dtype,
                   num_workers: int = 0) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for a specific phase (Train, Validation, or Test).

    :param data_folder: Path to the root data folder.
    :param model_configuration: Model configuration dictionary.
    :param split_path: Path to the JSON file defining the dataset split.
    :param phase: The phase for which to get the dataloader ('Train', 'Validation', or 'Test').
    :param device: The target torch.device.
    :param dtype: The target torch.dtype.
    :param num_workers: Number of worker threads for data loading (default 0).
    :return: A PyTorch DataLoader configured for the specified phase.
    """
    config_key = f"{phase}DataConfiguration"
    if config_key not in model_configuration:
        raise KeyError(f"Configuration for phase '{phase}' not found in model_configuration. "
                       f"Expected key: '{config_key}'")

    phase_configuration = model_configuration[config_key]

    dataloader = get_protein_dataloader(
        data_folder=data_folder,
        split_file_path=split_path,
        residue_crop_size=phase_configuration['residue_crop_size'],
        emphasize_beginning_crops=phase_configuration['emphasize_beginning_crops'],
        acceptance_slope_start=phase_configuration['acceptance_slope_start'],
        acceptance_slope_end=phase_configuration['acceptance_slope_end'],
        distribution_threshold=phase_configuration['distribution_threshold'],
        maximum_cluster_sequences=phase_configuration['maximum_cluster_sequences'],
        maximum_extra_msa_sequences=phase_configuration['maximum_extra_msa_sequences'],
        mask_probability=phase_configuration['mask_probability'],
        number_recycle_cycles=phase_configuration['number_recycle_cycles'],
        use_single_representative=phase_configuration['use_single_representative'],
        batch_size=phase_configuration['batch_size'],
        num_workers=num_workers,
        shuffle=phase_configuration['shuffle'],
        device=device,
        dtype=dtype)

    return dataloader


class PrecomputedProteinDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed and saved protein tensors.

    This dataset is used when the features (MSA clusters, ground truths, etc.)
    have already been extracted and saved as `.pt` files. It significantly
    speeds up the training process by avoiding repetitive feature extraction
    calculations.

    During training, it can randomly select from multiple precomputed samples
    (stochastic crops/masks) for each protein.
    """

    def __init__(self, precomputed_directory: str | Path, split_file_path: str | Path,
                 phase: str, existing_precomputed_samples: int):
        """
        Initializes the PrecomputedProteinDataset.

        Args:
            precomputed_directory (str | Path): Root directory where `.pt` files are stored.
            split_file_path (str | Path): Path to the split JSON file.
            phase (str): The current phase ('Train', 'Validation', or 'Test').
            existing_precomputed_samples (int): The number of stochastic samples
                available for each protein in the precomputed directory.
        """
        self.precomputed_directory = Path(precomputed_directory) / phase
        self.split_file_path = Path(split_file_path)
        self.existing_precomputed_samples = existing_precomputed_samples
        self.phase = phase

        # Load split
        cluster_mapping = read_json(str(self.split_file_path))
        self.protein_ids = []
        for members in cluster_mapping.values():
            self.protein_ids.extend(members)

    def __len__(self) -> int:
        """
        Returns the total number of proteins in the precomputed split.

        Returns:
            int: Total protein count.
        """
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Loads a precomputed tensor dictionary for a specific protein.

        Args:
            index (int): Index of the protein.

        Returns:
            dict[str, torch.Tensor]: The loaded features and ground truths.
                Shapes match those defined in `ProteinDataset.__getitem__`.
        """
        protein_id = self.protein_ids[index]

        # Randomly select one of the precomputed samples if in Train and Validation phase,
        # otherwise just take the first sample (0).
        if self.phase in ["Train", "Validation"]:
            sample_index = random.randint(0, self.existing_precomputed_samples - 1)
        else:
            sample_index = 0

        file_path = self.precomputed_directory / f"{protein_id}_sample_{sample_index}.pt"

        return torch.load(file_path)


def get_precomputed_dataloader(precomputed_directory: str | Path,
                               split_file_path: str | Path,
                               phase: str,
                               existing_precomputed_samples: int,
                               batch_size: int,
                               num_workers: int,
                               shuffle: bool) -> DataLoader:
    """
    Creates a DataLoader for precomputed protein data.

    This is the fastest way to stream data into the model, as it bypasses
    all heavy feature extraction (MSA clustering, distogram labeling) at runtime.

    Args:
        precomputed_directory (str | Path): Path to the precomputed data.
        split_file_path (str | Path): Path to the split JSON file.
        phase (str): 'Train', 'Validation', or 'Test'.
        existing_precomputed_samples (int): Available samples per protein.
        batch_size (int): Samples per batch.
        num_workers (int): Parallel worker threads.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A configured PyTorch DataLoader.
    """

    dataset = PrecomputedProteinDataset(precomputed_directory=precomputed_directory, split_file_path=split_file_path,
                                        phase=phase, existing_precomputed_samples=existing_precomputed_samples)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=protein_collate_fn, drop_last=False)

    return dataloader
