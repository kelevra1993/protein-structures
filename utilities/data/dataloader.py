import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utilities.data.input import ModelInput
from utilities.os_utilities import read_json


class ProteinDataset(Dataset):
    """
    A PyTorch Dataset for loading protein structures and their associated MSAs.

    This dataset reads a split file (e.g., Train.json or Validation.json) which
    contains clusters of proteins to load. It then constructs `ModelInput` objects
    for each protein found in the dataset folder.
    """

    def __init__(self, data_folder: str, split_file_path: str,
                 acceptance_slope_start: int = 256,
                 acceptance_slope_end: int = 512,
                 residue_crop_size: int = 256,
                 distribution_threshold: int = 80,
                 maximum_cluster_sequences: int = 512,
                 maximum_extra_msa_sequences: int = 5120,
                 mask_probability: float = 0.15,
                 number_recycle_cycles: int = 1,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        Initializes the ProteinDataset.

        :param data_folder: Path to the root data folder containing 'structures', 'records', and 'raw_msa'.
        :param split_file_path: Path to the JSON file defining the train/validation split (keys are clusters, values are lists of protein IDs).
        :param acceptance_slope_start: Threshold for input filtering probability calculation.
        :param acceptance_slope_end: Upper threshold for input filtering probability calculation.
        :param residue_crop_size: The number of residues to crop from each sequence during sampling.
        :param distribution_threshold: Threshold to filter out sequences with highly biased amino acid distributions.
        :param maximum_cluster_sequences: Max number of sequences for the main MSA clusters.
        :param maximum_extra_msa_sequences: Max number of sequences for the extra MSA stack.
        :param mask_probability: Probability of masking residues in MSA clusters.
        :param number_recycle_cycles: Number of recycling iterations to simulate in the batch (used by ModelInput.get_data).
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
        self.distribution_threshold = distribution_threshold
        self.maximum_cluster_sequences = maximum_cluster_sequences
        self.maximum_extra_msa_sequences = maximum_extra_msa_sequences
        self.mask_probability = mask_probability
        self.number_recycle_cycles = number_recycle_cycles

        # Parse the JSON split file
        # The structure is assumed to be: { "cluster_id": ["protein_id_1", "protein_id_2"], ... }
        cluster_mapping = read_json(path=str(self.split_file_path))

        # Flatten the clusters into a single list of protein IDs
        # Here we just take all of the protiens of all of the clusters
        # Todo we will later implement a key call use_single_representative, and when available,
        #  just take a random member and not all of them
        #  consider using shuffling as well after all of the members have been added to self.protein_ids ?
        self.protein_ids = []
        for cluster, members in cluster_mapping.items():
            self.protein_ids.extend(members)

    def __len__(self) -> int:
        """
        Returns the total number of proteins in this dataset split.
        """
        return len(self.protein_ids)

    def __getitem__(self, index: int) -> dict:
        """
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
            distribution_threshold=self.distribution_threshold,
            maximum_cluster_sequences=self.maximum_cluster_sequences,
            maximum_extra_msa_sequences=self.maximum_extra_msa_sequences,
            mask_probability=self.mask_probability,
            device=self.device,
            dtype=self.dtype)

        # Get the cropped and recycled batch data
        # NOTE : if self.residue_crop_size is None then we just get the whole data
        batch_data = model_input.get_data(number_samples=self.number_recycle_cycles,seed=None,batch_mode=False)

        return batch_data


def protein_collate_fn(batch: list[dict]) -> dict:
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


def get_protein_dataloaders(data_folder: str,
                            train_split_path: str,
                            validation_split_path: str,
                            batch_size: int = 1,
                            num_workers: int = 0,
                            **dataset_kwargs) -> tuple[DataLoader, DataLoader]:
    """
    # Todo later we will add all the kwargs so that they can be clearly defined.
    Creates and returns the training and validation PyTorch DataLoaders.

    :param data_folder: Path to the root data folder.
    :param train_split_path: Path to the JSON file for the training split.
    :param validation_split_path: Path to the JSON file for the validation split.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of worker threads for data loading.
    :param dataset_kwargs: Additional arguments passed to the ProteinDataset.
    :return: A tuple containing (train_dataloader, validation_dataloader).
    """

    train_dataset = ProteinDataset(
        data_folder=data_folder,
        split_file_path=train_split_path,
        **dataset_kwargs)

    validation_dataset = ProteinDataset(
        data_folder=data_folder,
        split_file_path=validation_split_path,
        **dataset_kwargs)

    # Use our custom collate function to handle varying sequence lengths
    # TODO Test the shuffling
    # TODO Test prefetch_factor to see how much we can accelerate it
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle clusters/proteins for training
        num_workers=num_workers,
        collate_fn=protein_collate_fn,
        drop_last=False)

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=protein_collate_fn,
        drop_last=False)

    return train_dataloader, validation_dataloader
