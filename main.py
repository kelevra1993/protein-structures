import torch
from pathlib import Path
from trainer.trainer import Trainer
from utilities.os_utilities import load_experiment_configuration


def main():
    # Define configuration path
    experiment_configuration_path = Path(__file__).parent / "configurations" / "experiment_configuration.yaml"
    
    # Load configuration in one line
    experiment_configurations = load_experiment_configuration(experiment_configuration_path)

    # Define experiment folder (where logs and weights are stored)
    experiment_folder = Path(experiment_configurations["experiment_parent_folder"]) / experiment_configurations["experiment_name"]
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # Extract absolute paths
    data_folder = Path(experiment_configurations["data_folder"])
    configuration_path = Path(experiment_configurations["configuration_path"])
    train_split = Path(experiment_configurations["train_split_file"])
    validation_split = Path(experiment_configurations["validation_split_file"])
    test_split = Path(experiment_configurations["test_split_file"])

    # Training parameters
    number_iterations = int(experiment_configurations["number_iterations"])
    weight_saving_iterations = experiment_configurations["weight_saving_iterations"]
    compute_validation_iteration = experiment_configurations["compute_validation_iteration"]
    learning_rate = float(experiment_configurations["learning_rate"])
    information_dump = experiment_configurations["information_dump"]
    resume_training = experiment_configurations["resume_training"]

    # Handle dtype
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map.get(experiment_configurations["dtype"], torch.float32)

    # Initialize Trainer
    # project_root is now the experiment_folder
    trainer = Trainer(project_root=experiment_folder, configuration_path=configuration_path, data_folder=data_folder,
                      train_split_file=train_split, validation_split_file=validation_split, test_split_file=test_split,
                      number_iterations=number_iterations, weight_saving_iterations=weight_saving_iterations,
                      compute_validation_iteration=compute_validation_iteration, learning_rate=learning_rate,
                      dtype=dtype, information_dump=information_dump, resume_training=resume_training)

    # Start training loop
    print(f"Starting Training for experiment: {experiment_configurations['experiment_name']}...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
