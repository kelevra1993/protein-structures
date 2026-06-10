import os
import time

import torch
import torch.optim as optim
from tomlkit import value
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
                 train_split_file: Path, validation_split_file: Path,
                 number_iterations: int, weight_saving_iterations: int,
                 compute_validation_iteration: bool,

                 learning_rate: float,

                 dtype: torch.dtype):
        # Get device and dtype
        self.device = get_device()
        self.dtype = dtype

        print(f"For Training We Will Be Using device: {self.device}")

        # Root where training will be performed
        # Will contain tensorboard logs
        # Will contain weights
        # Will contain model outputs for debugging
        self.project_root = project_root
        self.tensorboard_directory, self.weights_directory = self.setup_training_paths()

        # Setup writers
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Load Training Configuration containing all the training parameters
        self.configuration_path = configuration_path
        self.model_configuration = load_configuration(configuration_path=str(self.configuration_path))

        # Setting up dataloaders
        self.data_folder = data_folder
        self.train_split_file = train_split_file
        self.validation_split_file = validation_split_file

        # Setup training and validation dataloaders
        self.train_dataloader, self.validation_dataloader = get_train_and_validation_dataloader(
            train_data_folder=str(self.data_folder),
            validation_data_folder=str(self.data_folder),
            model_configuration=self.model_configuration,
            train_split_path=str(self.train_split_file),
            validation_split_path=str(self.validation_split_file),
            device=self.device,
            dtype=self.dtype)

        # Initialize Model and Optimizer
        self.model = Model(configuration=self.model_configuration, device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

        # Setting Up The Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # Iterations + Weight Saving + Validation computation (if true slows down training)
        self.training_iterations = number_iterations
        self.weight_saving_iterations = weight_saving_iterations
        self.compute_validation_iteration = compute_validation_iteration

    def setup_training_paths(self):
        """
        todo document
        :return:
        """
        tensorboard_directory = self.project_root / "Tensorboard"
        weights_directory = self.project_root / "Weights"

        # Create directories
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        weights_directory.mkdir(exist_ok=True, parents=True)

        return tensorboard_directory, weights_directory

    def setup_tensorboard_writers(self):
        """
        todo add documentation
        :return:
        """
        training_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Train"))
        if self.compute_validation_iteration:
            validation_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Validation"))
        else:
            validation_writer = None

        return training_writer, validation_writer

    def run_training_loop(self):
        # Convert to iterators
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        for training_iteration in range(1, self.training_iterations, 1):

            # Save the model
            if training_iteration % self.weight_saving_iterations == 0:
                self.save_model(iteration=training_iteration)

            # Get next training elements
            training_batch_dictionary = next(training_dataloader_iterator)

            # Set the module in training mode.
            # Reset the gradients of all optimized classes :`torch.Tensor` s.
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass containing batch dictionary and tensorboard logger
            training_loss = self.run_model_iteration(batch_input_dictionary=training_batch_dictionary,
                                                     writer=self.training_writer, iteration=training_iteration)

            # Backward and Step
            training_loss.backward()
            self.optimizer.step()

            # Validation phase : No gradient computation
            if self.compute_validation_iteration:
                with torch.no_grad():
                    # Get data
                    validation_batch_dictionary = next(validation_dataloader_iterator)

                    # Forward pass containing batch dictionary and tensorboard logger
                    validation_loss = self.run_model_iteration(
                        batch_input_dictionary=validation_batch_dictionary,
                        writer=self.validation_writer,
                        iteration=training_iteration)

        # Close all the summary writers
        self.training_writer.close()
        if self.compute_validation_iteration:
            self.validation_writer.close()

    def run_model_iteration(self, batch_input_dictionary, writer, iteration):
        """
        todo to be documented
        :param batch_input_dictionary:
        :return:
        """
        model_outputs = self.model(batch_input_dictionary=batch_input_dictionary)

        # Loss Calculation (Mean over cycles)
        fape_loss = model_outputs['overall_fape_loss'].mean()
        auxillary_loss = model_outputs['auxillary_loss'].mean()
        lddt_loss = model_outputs['predicted_lddt_loss'].mean()

        # Get Distogram Loss
        distogram_loss = model_outputs["distogram_loss"]

        # Get full training loss
        total_loss = fape_loss + auxillary_loss + lddt_loss + distogram_loss

        # Log data to tensorboard
        self.log_losses_to_tensorboard(writer=writer,
                                       iteration=iteration,
                                       total_loss=total_loss,
                                       fape_loss=fape_loss,
                                       auxillary_loss=auxillary_loss,
                                       lddt_loss=lddt_loss,
                                       distogram_loss=distogram_loss)

        return total_loss

    def save_model(self, iteration):
        """
        todo document
        :param iteration:
        :return:
        """
        model_directory = self.weights_directory / f"Iteration_{iteration}"
        print(f"Saving Model At : {model_directory}...")
        torch.save(self.model.state_dict(), model_directory / f"model_{iteration:06}.pt")
        print("Model Successfully Saved.")

    def log_losses_to_tensorboard(self, writer, iteration,
                                  total_loss, fape_loss, auxillary_loss, lddt_loss, distogram_loss):
        writer.add_scalar("Total Loss", total_loss.item(), iteration)
        writer.add_scalar("Frame Aligned Point Error Loss", fape_loss.item(), iteration)
        writer.add_scalar("Auxillary Loss", auxillary_loss.item(), iteration)
        writer.add_scalar("Local Distance Difference Test Loss", lddt_loss.item(), iteration)
        writer.add_scalar("Distogram Loss", distogram_loss.item(), iteration)


    def dump_training_information(self, iteration):
        pass
