from pathlib import Path
from trainer.trainer import Trainer
from utilities.os_utilities import load_experiment_configuration, load_configuration


def main():
    # Define configuration path
    experiment_configuration_path = Path(__file__).parent / "configurations" / "experiment_configuration.yaml"

    # Load configuration
    experiment_configuration = load_experiment_configuration(experiment_configuration_path)

    # Initialize Trainer
    trainer = Trainer(
        project_root=experiment_configuration["project_root"],
        model_configuration=load_configuration(experiment_configuration["configuration_path"]),
        data_folder=experiment_configuration["data_folder"],
        train_split_file=experiment_configuration["train_split_file"],
        validation_split_file=experiment_configuration["validation_split_file"],
        test_split_file=experiment_configuration["test_split_file"],
        number_iterations=experiment_configuration["number_iterations"],
        weight_saving_iterations=experiment_configuration["weight_saving_iterations"],
        compute_validation_iteration=experiment_configuration["compute_validation_iteration"],
        learning_rate=experiment_configuration["learning_rate"],
        dtype=experiment_configuration["dtype"],
        information_dump=experiment_configuration["information_dump"],
        resume_training=experiment_configuration["resume_training"])

    # Start training loop
    print(f"Starting Training for experiment: {experiment_configuration['experiment_name']}...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
