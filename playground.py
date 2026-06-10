import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from full_model.model import Model
from utilities.os_utilities import load_configuration
from utilities.tensor_utilities import get_device, print_tensor_shape, print_tensor_list
from utilities.data.dataloader import get_protein_dataloader, get_train_and_validation_dataloader
from utilities.loss_utilities import compute_distogram_loss
from utilities.constants import atom_types


class Trainer():

    def __init__(self, project_root: Path, configuration_path: Path, data_folder: Path,
                 train_split_file: Path, validation_split_file: Path, dtype: torch.dtype):

        # Get device and dtype
        self.device = get_device()
        self.dtype = dtype

        print(f"For Training We Will Be Using device: {self.device}")

        # Root where training will be performed
        # Will contain tensorboard logs
        # Will contain weights
        # Will contain model outputs for debugging
        self.project_root = project_root

        # Load Training Configuration containing all the training parameters
        self.configuration_path = configuration_path
        self.model_configuration = load_configuration(configuration_path=str(self.configuration_path))

        # Setting up dataloaders
        self.data_folder = data_folder
        self.train_split_file = train_split_file
        self.validation_split_file = validation_split_file


    def setup_training_paths(self):

        tensorboard_directory = self.project_root / "Tensorboard"
        weights_directory = self.project_root / "Weights"

