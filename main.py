import torch
from pathlib import Path
from trainer.trainer import Trainer


def main():
    # Define project paths
    project_root = Path(__file__).parent
    data_folder = project_root / "data_examples" / "openfold"
    configuration_path = project_root / "configurations" / "tiny_configuration.yaml"

    # Define dataset splits
    train_split = project_root / "dataset_splits" / "Train_small.json"
    validation_split = project_root / "dataset_splits" / "Validation_small.json"
    test_split = project_root / "dataset_splits" / "Test_small.json"

    # Training parameters
    number_iterations = 501  # To include the 500th iteration check
    weight_saving_iterations = 50
    compute_validation_iteration = True
    learning_rate = 1e-4
    dtype = torch.float32
    information_dump = 10

    # Initialize Trainer
    trainer = Trainer(project_root=project_root, configuration_path=configuration_path, data_folder=data_folder,
                      train_split_file=train_split, validation_split_file=validation_split, test_split_file=test_split,
                      number_iterations=number_iterations, weight_saving_iterations=weight_saving_iterations,
                      compute_validation_iteration=compute_validation_iteration, learning_rate=learning_rate,
                      dtype=dtype, information_dump=information_dump, resume_training=False)

    # Start training loop
    print("Starting Training Example...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
