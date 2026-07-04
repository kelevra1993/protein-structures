import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional

from full_model.model import Model
from utilities.os_utilities import load_configuration, print_red, print_green, print_blue, print_yellow, to_modelcif
from utilities.tensor_utilities import get_device, print_tensor_status, print_tensor_list, extract_angles
from utilities.data.dataloader import get_dataloader, get_precomputed_dataloader
from utilities.constants import index_to_x, index_to_xxx, chi_angles_mask


class Trainer:
    """
    The Trainer class orchestrates the entire training lifecycle of the AlphaFold II model.
    It manages data loading, model initialization, optimization, validation, checkpointing,
    and logging to TensorBoard.

    This class serves as the central hub for executing training runs, ensuring that all 
    components (model, data, optimizer) are correctly coordinated and that progress 
    is tracked systematically.
    """

    def __init__(self, project_root: Path, model_configuration: Dict[str, Any], data_folder: Path,
                 train_split_file: Path, validation_split_file: Path, test_split_file: Path,
                 number_iterations: int, weight_saving_iterations: int,
                 compute_validation_iteration: bool,
                 learning_rate: float,
                 dtype: torch.dtype,
                 information_dump: int,
                 resume_training: bool,
                 precompute_data: bool,
                 experiment_name: str,
                 precomputed_samples: int):
        """
        Initializes the Trainer with all necessary components for a training run.

        Args:
            project_root (Path): The root directory where training outputs (weights, logs) will be saved.
            model_configuration (Dict[str, Any]): Dictionary containing model and data parameters.
            data_folder (Path): Root directory containing the processed protein data (structures, records, msa).
            train_split_file (Path): Path to the JSON file defining the training dataset split.
            validation_split_file (Path): Path to the JSON file defining the validation dataset split.
            test_split_file (Path): Path to the JSON file defining the test dataset split.
            number_iterations (int): Total number of training iterations to perform.
            weight_saving_iterations (int): Interval (in iterations) at which to save model weights.
            compute_validation_iteration (bool): If True, performs a validation step during each training iteration.
            learning_rate (float): Initial learning rate for the Adam optimizer.
            dtype (torch.dtype): Data type to be used for all model tensors (e.g., torch.float32 or torch.float64).
            information_dump (int): Interval at which rolling average metrics are printed to the console.
            resume_training (bool): If True, restores the model and optimizer state from the last checkpoint.
            precompute_data (bool): Whether to use precomputed data loaders.
            experiment_name (str): The name of the experiment.
            precomputed_samples (int): Number of precomputed samples available per protein.
        """
        # Get device and dtype
        self.device = get_device()
        self.dtype = dtype

        print(f"For Training We Will Be Using device: {self.device}")

        # Basic training parameters
        self.project_root = project_root
        self.training_iterations = number_iterations
        self.weight_saving_iterations = weight_saving_iterations
        self.compute_validation_iteration = compute_validation_iteration
        self.information_dump = information_dump
        self.learning_rate = learning_rate

        # Root where training will be performed
        # Will contain tensorboard logs
        # Will contain weights
        # Will contain model outputs for debugging
        self.tensorboard_directory, self.weights_directory = self.setup_training_paths()

        # Setup writers
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Load Training Configuration containing all the training parameters
        self.model_configuration = model_configuration

        # Setting up dataloaders
        self.data_folder = data_folder
        self.train_split_file = train_split_file
        self.validation_split_file = validation_split_file
        self.test_split_file = test_split_file

        self.precompute_data = precompute_data
        self.precomputed_directory = self.data_folder / experiment_name if self.precompute_data else None
        self.precomputed_samples = precomputed_samples

        # Get data loaders depending on the whether we used precomputation or not.
        self.train_dataloader, self.validation_dataloader, self.test_dataloader = self.get_trainer_data_loaders()

        # Initialize Model and Optimizer
        self.model = Model(configuration=self.model_configuration, device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

        # Setting Up The Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # Restoration logic
        self.start_iteration = 1
        if resume_training:
            self.start_iteration = self.restore_last_model()

        # Metric names mapping for logging and tracking
        self.tracked_metrics_mapping = {
            "total_loss": "Total Loss",
            "fape_loss": "Frame Aligned Point Error Loss",
            "auxillary_loss": "Auxillary Loss",
            "lddt_loss": "Local Distance Difference Test Loss",
            "distogram_loss": "Distogram Loss",
            "lddt_metric": "True lDDT Metric",
            "unclamped_fape": "Unclamped Physical FAPE Metric"
        }

        # Print Model Size
        self.print_model_size()

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

    def get_trainer_data_loaders(self):
        """
        Initializes and returns the dataloaders for the training, validation, and test phases.

        This method acts as a factory, deciding whether to load the standard runtime
        dataloaders (which perform MSA clustering, cropping, and feature extraction dynamically)
        or the precomputed dataloaders (which load pre-processed `.pt` tensors directly from disk
        for maximum training speed), based on the `precompute_data` flag.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing:
                - train_dataloader: The dataloader for the training split.
                - validation_dataloader: The dataloader for the validation split.
                - test_dataloader: The dataloader for the test split.
        """

        # Setup training, validation, and test dataloaders
        if self.precompute_data:
            print_blue("Using Precomputed DataLoaders.", add_separators=True)
            train_dataloader = get_precomputed_dataloader(
                precomputed_directory=str(self.precomputed_directory), split_file_path=str(self.train_split_file),
                phase="Train", existing_precomputed_samples=self.precomputed_samples,
                batch_size=self.model_configuration["TrainDataConfiguration"]["batch_size"],
                num_workers=0, shuffle=self.model_configuration["TrainDataConfiguration"]["shuffle"])

            validation_dataloader = get_precomputed_dataloader(
                precomputed_directory=str(self.precomputed_directory), split_file_path=str(self.validation_split_file),
                phase="Validation", existing_precomputed_samples=self.precomputed_samples,
                batch_size=self.model_configuration["ValidationDataConfiguration"]["batch_size"],
                num_workers=0, shuffle=self.model_configuration["ValidationDataConfiguration"]["shuffle"])

            test_dataloader = get_precomputed_dataloader(
                precomputed_directory=str(self.precomputed_directory), split_file_path=str(self.test_split_file),
                phase="Test", existing_precomputed_samples=self.precomputed_samples,
                batch_size=self.model_configuration["TestDataConfiguration"]["batch_size"],
                num_workers=0, shuffle=self.model_configuration["TestDataConfiguration"]["shuffle"])
        else:
            print_yellow("Using Standard RunTime DataLoaders.", add_separators=True)
            train_dataloader = get_dataloader(
                data_folder=str(self.data_folder),
                model_configuration=self.model_configuration,
                split_path=str(self.train_split_file),
                phase="Train",
                device=self.device,
                dtype=self.dtype)

            validation_dataloader = get_dataloader(
                data_folder=str(self.data_folder),
                model_configuration=self.model_configuration,
                split_path=str(self.validation_split_file),
                phase="Validation",
                device=self.device,
                dtype=self.dtype)

            test_dataloader = get_dataloader(
                data_folder=str(self.data_folder),
                model_configuration=self.model_configuration,
                split_path=str(self.test_split_file),
                phase="Test",
                device=self.device,
                dtype=self.dtype)

        return train_dataloader, validation_dataloader, test_dataloader

    def set_input_dictionary_device(self, input_dictionary: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Moves all tensors within the input feature dictionary to the target computation device.

        This ensures that all input data (features, ground truth labels, masks) 
        are loaded onto the correct computation device (e.g., CPU, CUDA, MPS) prior 
        to being passed to the model for forward propagation, keeping it consistent 
        with the model's location.

        Args:
            input_dictionary (Dict[str, torch.Tensor]): A dictionary containing input features and labels.
                Contains tensors with shapes such as:
                - input_msa_feature: (batch_size, number_clusters, number_residues, msa_feature_dimension, number_cycles)
                - input_extra_msa_feature: (batch_size, number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_cycles)
                - input_sequence_feature: (batch_size, number_residues, input_sequence_feature_dimension)
                - input_residue_index_feature: (batch_size, number_residues)
                - ground_truth_frames: (batch_size, number_residues, 8, 4, 4)
                - ground_truth_angles: (batch_size, number_residues, 7, 2)
                - ground_truth_global_positions: (batch_size, number_residues, 37, 3)
                - distogram_labels: (batch_size, number_residues, number_residues)

        Returns:
            Dict[str, torch.Tensor]: The same dictionary with all its tensor values moved to
                the active computation device.
        """
        for key in list(input_dictionary.keys()):
            input_dictionary[key] = input_dictionary[key].to(self.device)

        return input_dictionary

    def set_input_dictionary_dtype(self, input_dictionary):
        """
        Casts specific continuous-valued tensors within the input dictionary to the configured dtype.

        While discrete indices (like residue or token indices) remain as integers, continuous 
        features and coordinates must be explicitly cast to the model's configured floating-point 
        type (e.g., torch.float32, torch.float64) to ensure cross-platform compatibility and 
        prevent type mismatch errors during computation. The 'sequence_labels' are explicitly 
        cast to torch.int64.

        Args:
            input_dictionary (Dict[str, torch.Tensor]): A dictionary containing input features and labels.
                Contains tensors with shapes such as:
                - input_msa_feature: (batch_size, number_clusters, number_residues, msa_feature_dimension, number_cycles)
                - input_extra_msa_feature: (batch_size, number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_cycles)
                - input_sequence_feature: (batch_size, number_residues, input_sequence_feature_dimension)
                - ground_truth_global_positions: (batch_size, number_residues, 37, 3)
                - ground_truth_frames: (batch_size, number_residues, 8, 4, 4)
                - ground_truth_angles: (batch_size, number_residues, 7, 2)
                - alternative_ground_truth_global_positions: (batch_size, number_residues, 37, 3)
                - alternative_ground_truth_frames: (batch_size, number_residues, 8, 4, 4)
                - alternative_ground_truth_angles: (batch_size, number_residues, 7, 2)
                - sequence_labels: (batch_size, number_residues)

        Returns:
            Dict[str, torch.Tensor]: The dictionary with targeted tensors cast to the correct dtype.
                Output tensor shapes remain identical to the input tensor shapes.
        """
        for key in ["input_msa_feature", "input_extra_msa_feature", "input_sequence_feature",
                    "ground_truth_global_positions", "ground_truth_frames", "ground_truth_angles",
                    "alternative_ground_truth_global_positions", "alternative_ground_truth_frames",
                    "alternative_ground_truth_angles"]:
            input_dictionary[key] = input_dictionary[key].to(self.dtype)

        input_dictionary["sequence_labels"] = input_dictionary["sequence_labels"].to(torch.int64)
        input_dictionary["distogram_labels"] = input_dictionary["distogram_labels"].to(torch.int64)

        return input_dictionary

    def run_benchmarking_training_loop(self, number_iterations=100000):
        """
        Executes a high-repetition loop to benchmark the model's performance.

        This method is designed for performance profiling and benchmarking. It repeatedly
        performs a single forward pass (via `run_model_iteration`) for a specified number
        of iterations, utilizing only the training dataloader. Unlike the standard
        `run_training_loop`, it does not perform backpropagation, optimizer updates,
        validation steps, or weight saving, focusing solely on the raw execution
        speed of the model's forward pass and loss calculation logic.

        The loop will automatically exit the application upon completion.

        Args:
            number_iterations (int): The total number of benchmark iterations to perform.
                Defaults to 100,000.
        """
        print(f"Running Training Benchmark On {number_iterations}")
        for _ in tqdm(range(number_iterations)):
            try:
                # Convert to iterators
                training_dataloader_iterator = iter(self.train_dataloader)

                # Initialize trackers
                training_trackers = self.get_metric_trackers()

                training_batch_dictionary = next(training_dataloader_iterator)
                _, _ = self.run_model_iteration(batch_input_dictionary=training_batch_dictionary,
                                                writer=self.training_writer,
                                                iteration=1,
                                                tracker_dictionary=training_trackers)
            except StopIteration:
                training_dataloader_iterator = iter(self.train_dataloader)
        print("Training Benchmark Completed.")
        exit()

    def run_training_loop(self):
        """
        Executes the main training loop for the specified number of iterations.
        
        This method iterates over the training dataloader, performs forward and backward passes,
        updates model parameters, periodically saves the model, and optionally runs validation steps.
        """
        # Convert to iterators
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        # Initialize trackers
        # Tracker dictionary example:
        # tracker = {"start_time": 123456789.0, "total_loss": 0.0, "fape_loss": 0.0,
        #           "auxillary_loss": 0.0, "lddt_loss": 0.0, "distogram_loss": 0.0}
        training_trackers = self.get_metric_trackers()
        if self.compute_validation_iteration:
            validation_trackers = self.get_metric_trackers()
        else:
            validation_trackers = None

        training_iteration = self.start_iteration

        # Periodic checkpointing for safety (backup)
        # We save the model every 1/4 of the weight_saving_iterations to prevent significant
        # data loss in case of an unexpected shutdown.
        backup_interval = max(1, self.weight_saving_iterations // 4)

        try:
            for training_iteration in range(self.start_iteration, self.training_iterations + self.start_iteration, 1):

                # Periodic checkpointing for safety (backup)
                if training_iteration % backup_interval == 0:
                    self.save_model(iteration=training_iteration)

                    # Only run the full test evaluation at the main weight_saving_iterations interval
                    if training_iteration % self.weight_saving_iterations == 0:
                        self.run_test_evaluation(iteration=training_iteration)

                # Get next training elements
                # Ensure that in case of stopIteration, just relaunch iterator
                try:
                    training_batch_dictionary = next(training_dataloader_iterator)
                except StopIteration:
                    training_dataloader_iterator = iter(self.train_dataloader)
                    training_batch_dictionary = next(training_dataloader_iterator)
                except FileNotFoundError as e:
                    continue

                # Set the module in training mode.
                # Reset the gradients of all optimized classes :`torch.Tensor` s.
                self.model.train()
                self.optimizer.zero_grad()

                # Forward pass containing batch dictionary and tensorboard logger
                training_loss, training_model_outputs = self.run_model_iteration(
                    batch_input_dictionary=training_batch_dictionary,
                    writer=self.training_writer,
                    iteration=training_iteration,
                    tracker_dictionary=training_trackers)

                # Backward and Step
                training_loss.backward()
                self.optimizer.step()

                # Validation phase : No gradient computation
                if self.compute_validation_iteration:
                    self.model.eval()
                    with torch.no_grad():
                        # Get data
                        # Ensure that in case of stopIteration, just relaunch iterator
                        try:
                            validation_batch_dictionary = next(validation_dataloader_iterator)
                        except StopIteration:
                            validation_dataloader_iterator = iter(self.validation_dataloader)
                            validation_batch_dictionary = next(validation_dataloader_iterator)
                        except FileNotFoundError as e:
                            continue
                        # Forward pass containing batch dictionary and tensorboard logger
                        _, _ = self.run_model_iteration(
                            batch_input_dictionary=validation_batch_dictionary,
                            writer=self.validation_writer,
                            iteration=training_iteration,
                            tracker_dictionary=validation_trackers)

                # Console log dump
                if training_iteration % self.information_dump == 0:
                    training_trackers = self.console_log_update_tracker(
                        iterations=training_iteration,
                        training_tracker_dictionary=training_trackers,
                        validation_tracker_dictionary=validation_trackers)

                    # Function that will be used to debug convergence
                    self.console_log_prediction_comparisons(training_model_outputs=training_model_outputs,
                                                            training_batch_dictionary=training_batch_dictionary)

                    if self.compute_validation_iteration:
                        validation_trackers = self.get_metric_trackers()

        except KeyboardInterrupt:
            print_red(f"\nTraining Interrupted by User at iteration {training_iteration}.", add_separators=True)

        except FileNotFoundError:
            print_red(f"\nFile Not Found at iteration {training_iteration}.", add_separators=True)

        finally:
            self.save_model(iteration=training_iteration)
            print_green(f"Model successfully saved at iteration {training_iteration}. Exiting Training.",
                        add_separators=True)
            # Close all the summary writers
            self.training_writer.close()
            if self.compute_validation_iteration:
                self.validation_writer.close()

    def run_test_evaluation(self, iteration: int):
        """
        Performs a full evaluation on the test dataset and logs results to a file.

        Args:
            iteration (int): The current training iteration index.
        """
        print(f"Starting Full Test Evaluation at Iteration {iteration}...")
        self.model.eval()
        test_trackers = {metric_key: 0.0 for metric_key in self.tracked_metrics_mapping.keys()}
        number_batches = len(self.test_dataloader)

        with torch.no_grad():
            for batch_input_dictionary in tqdm(self.test_dataloader, total=number_batches,
                                               desc=f"Test Evaluation Iteration {iteration}"):
                # Set input to the right device
                batch_input_dictionary = self.set_input_dictionary_device(input_dictionary=batch_input_dictionary)
                batch_input_dictionary = self.set_input_dictionary_dtype(input_dictionary=batch_input_dictionary)
                model_outputs = self.model(batch_input_dictionary=batch_input_dictionary)

                # Calculate metrics (mean over cycles)
                fape_loss = model_outputs['overall_fape_loss'].mean().item()
                auxillary_loss = model_outputs['auxillary_loss'].mean().item()
                lddt_loss = model_outputs['predicted_lddt_loss'].mean().item()
                distogram_loss = model_outputs["distogram_loss"].item()
                total_loss = fape_loss + auxillary_loss + lddt_loss + distogram_loss
                lddt_metric = model_outputs["true_lddt"].mean().item()
                unclamped_fape = model_outputs["unclamped_fape"].mean().item()

                # Accumulate
                test_trackers["total_loss"] += total_loss
                test_trackers["fape_loss"] += fape_loss
                test_trackers["auxillary_loss"] += auxillary_loss
                test_trackers["lddt_loss"] += lddt_loss
                test_trackers["distogram_loss"] += distogram_loss
                test_trackers["lddt_metric"] += lddt_metric
                test_trackers["unclamped_fape"] += unclamped_fape

        # Calculate means
        for metric_key in test_trackers.keys():
            test_trackers[metric_key] /= number_batches

        # Log to file
        evaluation_file = self.project_root / "test_evaluation_results.txt"
        with open(evaluation_file, "a") as f:
            f.write("+" + "-" * 50 + "+\n")
            f.write(f"| Iteration: {iteration:<38} |\n")
            f.write("+" + "-" * 50 + "+\n")
            for metric_key, display_name in self.tracked_metrics_mapping.items():
                f.write(f"| {display_name:<35} : {test_trackers[metric_key]:<8.4f} |\n")
            f.write("+" + "-" * 50 + "+\n\n")

        print(f"Full Test Evaluation Completed. Results appended to {evaluation_file}")

        # Run sample predictions for visualization
        self.run_sample_predictions(iteration=iteration, number_samples=50)

    def run_sample_predictions(self, iteration: int, number_samples: int = 50):
        """
        Runs inference on a subset of the test dataset and saves the predicted
        protein structures as ModelCIF files.

        This method facilitates visual inspection of the model's predictions by
        converting raw coordinate outputs into standard protein structure files (.cif).
        The files are saved in a dedicated 'test_sample_predictions' subdirectory
        within the current iteration's weight folder.

        Args:
            iteration (int): The current training iteration index, used for output organization.
            number_samples (int): The maximum number of test proteins to process for visualization.
                Defaults to 50.
        """
        print(f"Running {number_samples} Sample Predictions for Iteration {iteration}...")

        # Create output directory within the iteration's weight folder
        output_directory = self.weights_directory / f"Iteration_{iteration}" / "test_sample_predictions"
        output_directory.mkdir(exist_ok=True, parents=True)

        self.model.eval()
        samples_processed = 0

        # We iterate directly through the dataset to easily access protein IDs
        # and avoid complex collation for non-tensor metadata.
        dataset = self.test_dataloader.dataset

        with torch.no_grad():
            for i in range(min(number_samples, len(dataset))):
                protein_id = dataset.protein_ids[i]
                batch = dataset[i]

                # Add batch dimension and move to device/dtype
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}
                batch = self.set_input_dictionary_device(batch)
                batch = self.set_input_dictionary_dtype(batch)

                # Run inference
                model_outputs = self.model(batch_input_dictionary=batch)

                # Extract predicted positions and mask for the last cycle
                # final_positions shape: (1, number_residues, 37, 3, number_cycles)
                # position_mask shape: (1, number_residues, 37, number_cycles)
                predicted_positions = model_outputs["final_positions"][0, ..., -1]
                atom_mask = model_outputs["position_mask"][0, ..., -1]

                # Extract ground truth positions
                # shape: (1, number_residues, 37, 3)
                ground_truth_positions = batch["ground_truth_global_positions"][0, ...]

                # Extract sequence labels and convert back to single-letter amino acid list, replacing 'X' with 'UNK'
                # sequence_labels shape: (1, number_residues)
                sequence_indices = batch["sequence_labels"][0].cpu().numpy()
                sequence = [index_to_x[int(idx)] if index_to_x[int(idx)] != 'X' else 'UNK' for idx in sequence_indices]

                # Convert raw coordinates and mask to ModelCIF string format
                # passing both prediction and ground truth to be saved as Chain A and Chain B
                cif_string = to_modelcif(
                    atom_positions=predicted_positions,
                    atom_mask=atom_mask,
                    sequence=sequence,
                    description=f"AlphaFold II Prediction for {protein_id}",
                    ground_truth_positions=ground_truth_positions)

                # Save the structural model to disk (contains both chains)
                output_path = output_directory / f"{protein_id}.cif"
                with open(output_path, "w") as f:
                    f.write(cif_string)

                samples_processed += 1

        print_green(f"Successfully saved {samples_processed} sample predictions to {output_directory}")

    def run_model_iteration(self, batch_input_dictionary: Dict[str, torch.Tensor],
                            writer: SummaryWriter, iteration: int,
                            tracker_dictionary: Dict[str, Any] | None) -> torch.Tensor:
        """
        todo update docstring
        Performs a single forward pass of the model, calculates metrics, and logs metrics.

        Args:
            batch_input_dictionary (Dict[str, torch.Tensor]): A dictionary of input features.
                Key shapes:
                - input_msa_feature: (batch_size, number_clusters, number_residues, msa_feature_dimension, number_cycles)
                - input_extra_msa_feature: (batch_size, number_extra_sequences, number_residues, input_extra_msa_feature_dimension, number_cycles)
                - input_sequence_feature: (batch_size, number_residues, input_sequence_feature_dimension)
                - input_residue_index_feature: (batch_size, number_residues)
                - ground_truth_frames: (batch_size, number_residues, 8, 4, 4)
                - ground_truth_angles: (batch_size, number_residues, 7, 2)
                - ground_truth_global_positions: (batch_size, number_residues, 37, 3)
                - distogram_labels: (batch_size, number_residues, number_residues)
            writer (SummaryWriter): The TensorBoard writer to use for logging.
            iteration (int): The current training iteration index.
            tracker_dictionary (Dict[str, Any]): Dictionary to accumulate rolling average metrics.

        Returns:
            torch.Tensor: The total calculated loss for the current iteration (scalar).
        """
        # Set input to the right device
        batch_input_dictionary = self.set_input_dictionary_device(input_dictionary=batch_input_dictionary)
        batch_input_dictionary = self.set_input_dictionary_dtype(input_dictionary=batch_input_dictionary)

        # Run the model
        model_outputs = self.model(batch_input_dictionary=batch_input_dictionary)

        # Loss Calculation (Mean over cycles)
        fape_loss = model_outputs['overall_fape_loss'].mean()
        auxillary_loss = model_outputs['auxillary_loss'].mean()
        lddt_loss = model_outputs['predicted_lddt_loss'].mean()

        # Get Distogram Loss
        distogram_loss = model_outputs["distogram_loss"]

        # Get full training loss
        total_loss = fape_loss + auxillary_loss + lddt_loss + distogram_loss

        # Get the true lDDT metric (mean over residues and cycles)
        lddt_metric = model_outputs["true_lddt"].mean()

        # Get the physical FAPE metric (mean over cycles)
        unclamped_fape = model_outputs["unclamped_fape"].mean()

        # Store losses and other metrics in a dictionary
        metric_dictionary = {
            "total_loss": total_loss,
            "fape_loss": fape_loss,
            "auxillary_loss": auxillary_loss,
            "lddt_loss": lddt_loss,
            "distogram_loss": distogram_loss,
            "lddt_metric": lddt_metric,
            "unclamped_fape": unclamped_fape}

        # Update tracker dictionary (rolling average accumulation)
        for metric_key, metric_value in metric_dictionary.items():
            tracker_dictionary[metric_key] += metric_value.item() / self.information_dump

        # Log data to tensorboard (per-iteration)
        self.log_metrics_to_tensorboard(writer=writer,
                                        iteration=iteration,
                                        metric_dictionary=metric_dictionary)

        return total_loss, model_outputs

    def save_model(self, iteration: int):
        """
        Persists the current model weights and optimizer state to disk.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        model_directory = self.weights_directory / f"Iteration_{iteration}"
        model_directory.mkdir(exist_ok=True, parents=True)

        checkpoint_path = model_directory / f"model_{iteration:06}.pt"
        print(f"Saving Checkpoint At : {checkpoint_path}...")

        torch.save({
            'iteration': iteration,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, checkpoint_path)

        print("Checkpoint Successfully Saved.")

        # Update the full-checkpoint registry
        self.dump_in_checkpoint(iteration=iteration)

    def log_metrics_to_tensorboard(self, writer: SummaryWriter, iteration: int,
                                   metric_dictionary: Dict[str, torch.Tensor]):
        """
        Logs individual metric components and total loss to TensorBoard.

        Args:
            writer (SummaryWriter): The TensorBoard writer to use.
            iteration (int): The current iteration index.
            metric_dictionary (Dict[str, torch.Tensor]): Dictionary containing current iteration metrics.
        """
        for metric_key, display_name in self.tracked_metrics_mapping.items():
            writer.add_scalar(display_name, metric_dictionary[metric_key].item(), iteration)

    def run_profiling_loop(self, number_iterations: int = 10, wait: int = 1, warmup: int = 1,
                           active: int = 3, repeat: int = 2, record_shapes: bool = True,
                           profile_memory: bool = True, with_stack: bool = True):
        """
        Executes a localized training loop explicitly wrapped in the PyTorch Profiler.
        This captures hardware-level metrics (GPU/CPU execution time, memory allocation)
        and exports them to TensorBoard for performance bottleneck analysis.

        Args:
            number_iterations (int): Total iterations for the profiling loop.
            wait (int): Number of steps to skip before starting the profiler.
            warmup (int): Number of steps to run the profiler but discard data 
                          (allows caches/CUDA kernels to settle).
            active (int): Number of steps to actively record tracing data.
            repeat (int): Number of times to repeat the (wait -> warmup -> active) cycle.
            record_shapes (bool): If True, captures tensor shapes (helps identify memory hogs).
            profile_memory (bool): If True, tracks memory allocation/deallocation timelines.
            with_stack (bool): If True, records the Python call stack alongside operations.
        """
        print_blue(f"Starting Profiling Loop for {number_iterations} iterations...", add_separators=True)

        # Define where the profiler trace files (.pt.trace.json) should be saved.
        # These files are natively read by the TensorBoard Profiler plugin.
        profiler_directory = self.tensorboard_directory / "Profiler"
        profiler_directory.mkdir(exist_ok=True, parents=True)

        # Ensure model is in training mode
        self.model.train()
        training_dataloader_iterator = iter(self.train_dataloader)
        tracker_dictionary = self.get_metric_trackers()

        # Initialize the profiler context manager.
        # - schedule: Determines when the profiler is active based on step counts.
        # - on_trace_ready: A callback executed when a cycle completes; here we export to TensorBoard.
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_directory)),
                record_shapes=record_shapes, profile_memory=profile_memory, with_stack=with_stack) as profiler:

            for iteration in tqdm(range(1, number_iterations + 1), desc="Profiling Iterations"):

                try:
                    batch_input_dictionary = next(training_dataloader_iterator)
                except StopIteration:
                    training_dataloader_iterator = iter(self.train_dataloader)
                    batch_input_dictionary = next(training_dataloader_iterator)

                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass: We use the existing iteration method which handles device 
                # mapping and returns the calculated loss.
                training_loss, _ = self.run_model_iteration(batch_input_dictionary=batch_input_dictionary,
                                                            writer=self.training_writer,
                                                            iteration=iteration,
                                                            tracker_dictionary=tracker_dictionary)

                # Standard backward pass and optimizer step
                training_loss.backward()
                self.optimizer.step()

                # Crucial step: Tells the profiler that one logical step (iteration) has completed.
                # This advances the state machine (wait -> warmup -> active) defined in the schedule.
                profiler.step()

        print_green(f"Profiling completed. Trace data saved to: {profiler_directory}", add_separators=True)

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

    def console_log_prediction_comparisons(self, training_model_outputs: Dict[str, torch.Tensor],
                                           training_batch_dictionary: Dict[str, torch.Tensor],
                                           number_residues_to_consider: int = 5)-> None:
        """
        TODO To be documented
        """

        # Sequence of amino acid indices (batch=0)
        sequence_indices = training_batch_dictionary["sequence_labels"][0, :number_residues_to_consider].cpu().numpy()

        # Predictions: [batch=0, last_layer=-1, residues, angles, cos_sin, last_cycle=-1]
        predicted_angles = training_model_outputs["angles"][
            0, -1, :number_residues_to_consider, :, :, -1].detach().cpu().numpy()

        # Ground truth: [batch=0, residues, angles, cos_sin]
        true_angles = training_batch_dictionary["ground_truth_angles"][
            0, :number_residues_to_consider, :, :].cpu().numpy()

        filtered_predictions = []
        filtered_ground_truths = []
        amino_acid_labels = []
        angle_counts = []

        for i, amino_acid_index in enumerate(sequence_indices):
            # Amino acid string
            amino_acid_name = index_to_xxx.get(int(amino_acid_index), "UNK")
            amino_acid_labels.append(amino_acid_name)

            # The first 3 backbone angles are always present
            present_angles = [0, 1, 2]

            # Check chi angles
            chi_mask = chi_angles_mask[int(amino_acid_index)]
            for chi_index in range(4):
                if chi_mask[chi_index] == 1.0:
                    present_angles.append(chi_index + 3)

            angle_counts.append(len(present_angles))

            for angle_index in present_angles:
                filtered_predictions.append(predicted_angles[i, angle_index])
                filtered_ground_truths.append(true_angles[i, angle_index])

        if not filtered_predictions:
            return

        # Convert (cos, sin) to degrees using extract_angles
        # extract_angles returns shape (N, 1)
        predicted_degrees = extract_angles(filtered_predictions).flatten()
        true_degrees = extract_angles(filtered_ground_truths).flatten()

        # Compute delta normalized to [-180, 180]
        delta_deg = (predicted_degrees - true_degrees + 180) % 360 - 180

        # Formatting
        header_string = "Residues :"
        predicted_str = "Predicted:"
        true_string = "Expected :"
        delta_string = "Delta    :"

        angle_index = 0
        for count, amino_acid_name in zip(angle_counts, amino_acid_labels):
            # Each angle takes 7 characters: " +180 |"
            width = count * 7
            # Center the amino acid name in the width (minus the final '|' which is already included in the angle string, so width - 1)
            centered_aa = amino_acid_name.center(width - 1) + "|"
            header_string += " " + centered_aa

            for _ in range(count):
                predicted_str += f" {predicted_degrees[angle_index]:+04.0f} |"
                true_string += f" {true_degrees[angle_index]:+04.0f} |"
                delta_string += f" {delta_deg[angle_index]:+04.0f} |"
                angle_index += 1

        print("\n" + "=" * len(header_string))
        print(header_string)
        print("-" * len(header_string))
        print(predicted_str)
        print(true_string)
        print(delta_string)
        print("=" * len(header_string) + "\n")

    def extract_last_model_iteration(self) -> int:
        """
        Retrieves the iteration number of the most recently saved model checkpoint.

        This method reads the 'full-checkpoint' registry file located in the weights
        directory to determine the latest available checkpoint. This is crucial for
        seamlessly resuming training after an interruption without manual intervention.

        Returns:
            int: The iteration number of the last saved model, or 0 if no checkpoint exists.
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

    def restore_last_model(self, index_iteration: Optional[int] = None) -> int:
        """
        Restores the model and optimizer states to a specific or the most recent checkpoint.

        If `index_iteration` is provided, it restores that exact checkpoint (useful for
        running isolated evaluation or inference on a specific model state). If not provided,
        it automatically finds and loads the latest checkpoint to resume training.

        Args:
            index_iteration (Optional[int]): The specific iteration to restore. If None,
                it resolves to the last saved iteration.

        Returns:
            int: The iteration number from which training should commence. If a model was
                loaded, it returns `loaded_iteration + 1`. If no model was found, it returns 1.
        """

        if not index_iteration:
            index_iteration = self.extract_last_model_iteration()

        # Despite trying to get the last model None was found
        if not index_iteration:
            print_green(
                "No Initiation Model Weights Will Be Used..."
                "\nWe generate A New Model That Will Be Trained From Scratch.",
                add_separators=True)
            return 1
        else:
            self.load_model(iteration=index_iteration)
            print_green(f"We Loaded A Model That Was Previously Saved At Iteration {index_iteration}",
                        add_separators=True)
            # Resume from the next iteration
            return index_iteration + 1

    def load_model(self, iteration: int):
        """
        Loads the model weights and optimizer state from a specified iteration checkpoint.

        This method reads the serialized `.pt` file from disk and maps the tensors to
        the currently active device (CPU, CUDA, or MPS).

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """

        model_path = self.weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def dump_in_checkpoint(self, iteration: int):
        """
        Updates the checkpoint registry file to track the most recently saved model.

        The `full-checkpoint` file acts as a manifest, keeping a historical record of
        all saved checkpoints and explicitly marking the latest one. This ensures the
        resumption logic (`extract_last_model_iteration`) always knows where to start.

        Args:
            iteration (int): The iteration number of the newly saved checkpoint.
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

    def console_log_update_tracker(self, iterations: int,
                                   training_tracker_dictionary: Dict[str, Any],
                                   validation_tracker_dictionary: Optional[Dict[str, Any]] = None):
        """
        Prints the rolling average of metrics to the console.

        Args:
            iterations (int): Current global training iteration.
            training_tracker_dictionary (Dict[str, Any]): Rolling average trackers for training.
            validation_tracker_dictionary (Optional[Dict[str, Any]]): Rolling average trackers for validation.

        Returns:
            Dict[str, Any]: A fresh training tracker dictionary for the next interval.
        """
        print_length = 100
        print("-" * print_length)
        print(f"Iteration: {iterations}")

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            train_value = training_tracker_dictionary[metric_key]
            message = f"Moving Average of Training {display_name:40} : {train_value:.4f}"
            print_blue(message)

            if validation_tracker_dictionary is not None:
                validation_value = validation_tracker_dictionary[metric_key]
                validation_message = f"Moving Average of Validation {display_name:38} : {validation_value:.4f}"
                print_yellow(validation_message)

        duration = time.time() - training_tracker_dictionary['start_time']
        print(f"These {self.information_dump} iterations took {duration:.2f} seconds")
        print("-" * print_length)

        return self.get_metric_trackers()

    def get_metric_trackers(self) -> Dict[str, Any]:
        """
        Initializes a dictionary to track rolling averages of metrics.

        Returns:
            Dict[str, Any]: A dictionary with metric keys set to 0.0 and a start_time.
        """
        tracker_dictionary = {"start_time": time.time()}
        for metric_key in self.tracked_metrics_mapping.keys():
            tracker_dictionary[metric_key] = 0.0

        return tracker_dictionary
