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


def run_training():
    # 1. Setup Configuration and Device
    project_root = Path(__file__).parent
    configuration_path = project_root / "configurations" / "tiny_configuration.yaml"
    model_configuration = load_configuration(configuration_path=str(configuration_path))

    # TODO Check to see which device is available
    # TODO Maybe put this option in general options
    # Here we will add an option to check if mps or cuda and if not just cpu
    computer_device = torch.device("cpu")

    # TODO Maybe put this option in general options
    # Use float32 for training to be faster and compatible with most GPUs/MPS
    tensor_dtype = torch.float32

    print(f"Using device: {computer_device}")

    # Setup Dataloaders (Both train and validation point to validation split)
    data_folder = project_root / "data_examples" / "openfold"
    validation_split_path = project_root / "dataset_splits" / "Test.json"
    train_split_path = project_root / "dataset_splits" / "Test.json"

    # Setup Logging Directory
    logging_directory = "runs/alphafold_flash_training"

    # Setup Weights Path Directory and parameter file
    weights_dir = Path("model_weights")
    weights_dir.mkdir(exist_ok=True)

    train_dataloader, validation_dataloader = get_train_and_validation_dataloader(
        train_data_folder=str(data_folder),
        validation_data_folder=str(data_folder),
        model_configuration=model_configuration,
        train_split_path=str(train_split_path),
        validation_split_path=str(validation_split_path),
        device=computer_device,
        dtype=tensor_dtype)

    # Initialize Model and Optimizer
    alphafold_model = Model(configuration=model_configuration, device=computer_device, dtype=tensor_dtype)
    alphafold_model.to(device=computer_device, dtype=tensor_dtype)

    # Setting Up The Optimizer
    optimizer = optim.Adam(alphafold_model.parameters(), lr=1e-4)

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=logging_directory)

    # Distogram Bins
    distogram_bins = torch.linspace(2, 22, 64, device=computer_device, dtype=tensor_dtype)

    iteration_count = 0
    number_epochs = 100

    print("Starting training loop...")
    for epoch in range(number_epochs):
        alphafold_model.train()
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")

        for batch_dictionary in train_loop:
            optimizer.zero_grad()

            # Forward pass

            model_outputs = alphafold_model(batch_input_dictionary=batch_dictionary)

            # Loss Calculation (Mean over cycles)
            fape_loss = model_outputs['overall_fape_loss'].mean()
            aux_loss = model_outputs['auxillary_loss'].mean()
            lddt_loss = model_outputs['predicted_lddt_loss'].mean()

            # Get Distogram Loss
            distogram_loss = model_outputs["distogram_loss"]

            total_loss = fape_loss + aux_loss + lddt_loss + distogram_loss

            # Backward and Step
            total_loss.backward()
            optimizer.step()

            # Logging
            train_loop.set_postfix(loss=total_loss.item())
            writer.add_scalar("Loss/Train/Total", total_loss.item(), iteration_count)
            writer.add_scalar("Loss/Train/FAPE", fape_loss.item(), iteration_count)
            writer.add_scalar("Loss/Train/Aux", aux_loss.item(), iteration_count)
            writer.add_scalar("Loss/Train/LDDT", lddt_loss.item(), iteration_count)
            writer.add_scalar("Loss/Train/Distogram", distogram_loss.item(), iteration_count)

            iteration_count += 1

            # Weight Saving every 10,000 iterations
            if iteration_count % 10000 == 0:
                torch.save(alphafold_model.state_dict(), weights_dir / f"model_weights_iter_{iteration_count}.pt")

        # Validation phase (on the same data)
        alphafold_model.eval()
        val_loop = tqdm(validation_dataloader, desc=f"Epoch {epoch} [Val]")
        val_total_loss = 0.0

        with torch.no_grad():
            for batch_dictionary in val_loop:
                model_outputs = alphafold_model(batch_dictionary)

                fape_loss = model_outputs['overall_fape_loss'].mean()
                aux_loss = model_outputs['auxillary_loss'].mean()
                lddt_loss = model_outputs['predicted_lddt_loss'].mean()

                dist_loss = model_outputs["distogram_loss"]
                batch_loss = fape_loss + aux_loss + lddt_loss + dist_loss
                val_total_loss += batch_loss.item()
                val_loop.set_postfix(loss=batch_loss.item())

        avg_val_loss = val_total_loss / len(validation_dataloader)
        writer.add_scalar("Loss/Val/Total", avg_val_loss, epoch)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        # Save weights after each epoch
        torch.save(alphafold_model.state_dict(), weights_dir / f"model_weights_epoch_{epoch}.pt")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    run_training()
