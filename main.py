from pathlib import Path
from trainer.trainer import Trainer
from utilities.os_utilities import load_experiment_configuration, print_yellow
from utilities.data.data_precomputer.precompute_utilities import precompute_dataset
from utilities.tensor_utilities import get_device


def main():
    # Define configuration path
    experiment_configuration_path = Path(__file__).parent / "configurations" / f"{get_device()}_configuration.yaml"

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(experiment_configuration_path)

    # Conditionally execute data precomputation
    precompute_data = experiment_configuration.get("precompute_data", False)
    experiment_name = experiment_configuration.get("experiment_name", "experiment")
    precomputed_samples = experiment_configuration.get("precomputed_samples", 5)

    if precompute_data:
        output_directory = experiment_configuration["data_folder"] / experiment_name
        print_yellow(f"Precomputing Data Into {output_directory} With {precomputed_samples} Samples Per Protein...",
                     add_separators=True)
        precompute_dataset(experiment_configuration_path=str(experiment_configuration_path),
                           output_directory=str(output_directory),
                           number_samples=precomputed_samples)

    # Initialize Trainer
    trainer = Trainer(
        project_root=experiment_configuration["project_root"],
        model_configuration=model_configuration,
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
        resume_training=experiment_configuration["resume_training"],
        precompute_data=precompute_data,
        experiment_name=experiment_name,
        precomputed_samples=precomputed_samples)

    # Start training loop
    print(f"Starting Training For Experiment: {experiment_configuration['experiment_name']}...")
    trainer.run_training_loop()
    print("Training Example Completed.")


if __name__ == "__main__":
    main()
