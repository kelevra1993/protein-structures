import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from full_model.model import Model
from utilities.os_utilities import load_configuration, print_red, print_green, print_blue, print_yellow
from utilities.tensor_utilities import get_device, print_tensor_shape, print_tensor_list
from utilities.data.dataloader import get_protein_dataloader, get_train_and_validation_dataloader
from utilities.loss_utilities import compute_distogram_loss
from utilities.constants import atom_types


class Trainer:
    """
    The Trainer class orchestrates the entire training lifecycle of the AlphaFold II model.
    It manages data loading, model initialization, optimization, validation, checkpointing,
    and logging to TensorBoard.

    This class serves as the central hub for executing training runs, ensuring that all 
    components (model, data, optimizer) are correctly coordinated and that progress 
    is tracked systematically.
    """

    def __init__(self, project_root: Path, configuration_path: Path, data_folder: Path,
                 train_split_file: Path, validation_split_file: Path,
                 number_iterations: int, weight_saving_iterations: int,
                 compute_validation_iteration: bool,
                 learning_rate: float,
                 dtype: torch.dtype,
                 compute_model_size: bool = False):
        """
        Initializes the Trainer with all necessary components for a training run.

        Args:
            project_root (Path): The root directory where training outputs (weights, logs) will be saved.
            configuration_path (Path): Path to the YAML configuration file defining model and data parameters.
            data_folder (Path): Root directory containing the processed protein data (structures, records, msa).
            train_split_file (Path): Path to the JSON file defining the training dataset split.
            validation_split_file (Path): Path to the JSON file defining the validation dataset split.
            number_iterations (int): Total number of training iterations to perform.
            weight_saving_iterations (int): Interval (in iterations) at which to save model weights.
            compute_validation_iteration (bool): If True, performs a validation step during each training iteration.
            learning_rate (float): Initial learning rate for the Adam optimizer.
            dtype (torch.dtype): Data type to be used for all model tensors (e.g., torch.float32 or torch.float64).
            compute_model_size (bool): If True, prints the model size and exits.
        """
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
        self.compute_model_size = compute_model_size

    def setup_training_paths(self):
        """
        Sets up the directory structure for training outputs, including TensorBoard logs and model weights.

        Returns:
            Tuple[Path, Path]: A tuple containing the paths to the TensorBoard directory and the weights directory.
        """
        tensorboard_directory = self.project_root / "Tensorboard"
        weights_directory = self.project_root / "Weights"

        # Create directories
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        weights_directory.mkdir(exist_ok=True, parents=True)

        return tensorboard_directory, weights_directory

    def setup_tensorboard_writers(self):
        """
        Initializes TensorBoard SummaryWriters for logging training and validation metrics.

        Returns:
            Tuple[SummaryWriter, Optional[SummaryWriter]]: A tuple containing the training writer 
            and the validation writer (if validation is enabled).
        """
        training_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Train"))
        if self.compute_validation_iteration:
            validation_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Validation"))
        else:
            validation_writer = None

        return training_writer, validation_writer

    def run_training_loop(self):
        """
        Executes the main training loop for the specified number of iterations.
        
        This method iterates over the training dataloader, performs forward and backward passes,
        updates model parameters, periodically saves the model, and optionally runs validation steps.
        """
        # Convert to iterators
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        for training_iteration in range(1, self.training_iterations, 1):

            # Save the model
            if training_iteration % self.weight_saving_iterations == 0:
                self.save_model(iteration=training_iteration)

            # Get next training elements
            # Ensure that in case of stopIteration, just relaunch iterator
            try:
                training_batch_dictionary = next(training_dataloader_iterator)
            except StopIteration:
                training_dataloader_iterator = iter(self.train_dataloader)
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
                    # Ensure that in case of stopIteration, just relaunch iterator
                    try:
                        validation_batch_dictionary = next(validation_dataloader_iterator)
                    except StopIteration:
                        validation_dataloader_iterator = iter(self.validation_dataloader)
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

    def run_model_iteration(self, batch_input_dictionary: Dict[str, torch.Tensor],
                            writer: SummaryWriter, iteration: int) -> torch.Tensor:
        """
        Performs a single forward pass of the model, calculates losses, and logs metrics.

        Args:
            batch_input_dictionary (Dict[str, torch.Tensor]): A dictionary of input features.
                Key shapes:
                - input_msa_feature: (batch_size, number_clusters, number_residues, msa_feature_dimension, number_cycles)
                - input_extra_msa_feature: (batch_size, number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_cycles)
                - input_sequence_feature: (batch_size, number_residues, input_sequence_feature_dimension, number_cycles)
                - input_residue_index_feature: (batch_size, number_residues, number_cycles)
                - ground_truth_frames: (batch_size, number_residues, 8, 4, 4, number_cycles)
                - ground_truth_angles: (batch_size, number_residues, 7, 2, number_cycles)
                - ground_truth_global_positions: (batch_size, number_residues, 37, 3, number_cycles)
                - distogram_labels: (batch_size, number_residues, number_residues, number_cycles)
            writer (SummaryWriter): The TensorBoard writer to use for logging.
            iteration (int): The current training iteration index.

        Returns:
            torch.Tensor: The total calculated loss for the current iteration (scalar).
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

    def save_model(self, iteration: int):
        """
        Persists the current model weights to disk.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        model_directory = self.weights_directory / f"Iteration_{iteration}"
        model_directory.mkdir(exist_ok=True, parents=True)

        print(f"Saving Model At : {model_directory}...")
        torch.save(self.model.state_dict(), model_directory / f"model_{iteration:06}.pt")
        print("Model Successfully Saved.")

        # Update the full-checkpoint registry
        self.dump_in_checkpoint(iteration=iteration)

    def log_losses_to_tensorboard(self, writer: SummaryWriter, iteration: int,
                                  total_loss: torch.Tensor, fape_loss: torch.Tensor,
                                  auxillary_loss: torch.Tensor, lddt_loss: torch.Tensor,
                                  distogram_loss: torch.Tensor):
        """
        Logs individual loss components and total loss to TensorBoard.

        Args:
            writer (SummaryWriter): The TensorBoard writer to use.
            iteration (int): The current iteration index.
            total_loss (torch.Tensor): Total combined loss (scalar).
            fape_loss (torch.Tensor): Frame Aligned Point Error loss (scalar).
            auxillary_loss (torch.Tensor): Auxiliary head loss (scalar).
            lddt_loss (torch.Tensor): Local Distance Difference Test loss (scalar).
            distogram_loss (torch.Tensor): Distogram prediction loss (scalar).
        """
        writer.add_scalar("Total Loss", total_loss.item(), iteration)
        writer.add_scalar("Frame Aligned Point Error Loss", fape_loss.item(), iteration)
        writer.add_scalar("Auxillary Loss", auxillary_loss.item(), iteration)
        writer.add_scalar("Local Distance Difference Test Loss", lddt_loss.item(), iteration)
        writer.add_scalar("Distogram Loss", distogram_loss.item(), iteration)

    def dump_training_information(self, iteration: int):
        """
        Dumps additional training metadata or intermediate results for debugging.
        Currently a placeholder for future implementation.

        Args:
            iteration (int): The current iteration index.
        """
        pass

    # Functions that have been added
    def print_model_size(self):
        """
        Prints the estimated size of the model and optionally stops training if specified.
        """
        # Calculate model parameter size
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        # Calculate model buffer size
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        # Total model size in megabytes
        size_all_mb = (param_size + buffer_size) / 1024 ** 2

        # Print model size
        message = f"Estimated Model Size Without Optimizer : {size_all_mb:.2f} MB"
        print_yellow(message, add_separators=True)

        # Stop training message
        if self.compute_model_size:
            print_red("To Continue Training or Inference: Please Set compute_model_size To False",
                      add_separators=True)
            exit()

    def extract_last_model_iteration(self):
        """
        Function that is used to get the last model that was saved for a given run. Useful when we are resuming
        training.
        :return:
        """
        last_iteration = 0

        full_checkpoint_logger = self.weights_directory / "full-checkpoint"

        if not full_checkpoint_logger.exists():
            return last_iteration

        with open(full_checkpoint_logger, 'r') as file:
            first_line = file.readline().strip()

        last_iteration_string = first_line.split('"')[1]
        last_iteration = int(last_iteration_string.split("_")[-1])

        return last_iteration

    def restore_last_model(self, index_iteration=None):
        """
        Function that is used to restore the last model that was saved, if we are resuming training.
        If we are dealing with inference for validation or test set, to restore the model at index_iteration for evaluation.
        :param index_iteration: (int) iteration of interest for the model that we would like to restore
        :return:
        """

        if not index_iteration:
            index_iteration = self.extract_last_model_iteration()

        # Despite trying to get the last model None was found
        if not index_iteration:
            print_green(
                "No Initiation Model Weights Will Be Used...\nWe generate A New Model That Will Be Trained From Scratch.",
                add_separators=True)
        else:

            self.load_model(iteration=index_iteration)
            print_green(f"We Loaded A Model That Was Previously Saved At Iteration {index_iteration}",
                        add_separators=True)
        return index_iteration

    def load_model(self, iteration):
        """
        Function that is used to restore a model that was previously saved, either in order to continue
        training or to run inference.
        :param iteration: (int) iteration at which the model is being saved.
        :return:
        """

        model_path = self.weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"

        self.model.load_state_dict(torch.load(model_path))

    def dump_in_checkpoint(self, iteration):
        """
        Function that is used to dump information about the saved checkpoint in a file for later retrieval.
        :param iteration: (int) iteration at which the model is being saved.
        :return:
        """
        checkpoint_file = self.weights_directory / "full-checkpoint"

        # Saved the iteration in a checkpoint file
        try:
            with open(checkpoint_file, "r") as f:
                d = f.readlines()
        except FileNotFoundError:
            d = []

        with open(checkpoint_file, "w") as f:
            if len(d) == 0:
                d.append(f'model_checkpoint_path: "Iteration_{iteration}"\n')
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            else:
                d[0] = f'model_checkpoint_path: "Iteration_{iteration}"\n'
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            for line in d:
                f.write(line)
