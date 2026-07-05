"""
Hyperparameter Sweep Controller

This script acts as the automated controller for finding the optimal hyperparameters 
to rapidly overfit the network on a small dataset (memorization test). By systematically 
tuning hyperparameters, developers can debug training convergence efficiently. 
It uses the Optuna library to execute Bayesian optimization over a defined search space.
For each trial, it modifies the global `cuda_configuration.yaml`, invokes the main 
training loop, and assesses the model based on the final mean distance delta.
"""

import os
import shutil
import yaml
import json
import optuna
from pathlib import Path
import sys
import torch
import numpy as np
import random
from typing import Any

# Add project root to sys.path so we can import main
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import main

# ==========================================
# CONSTANTS & CONFIGURATION FOR SWEEP
# ==========================================
TOTAL_ITERATIONS = 750
INFORMATION_DUMP_FREQUENCY = 25

# Sweep configurations
DATABASE_URL = "sqlite:///sweep_results.db"
STUDY_NAME = "overfit_sweep"

# Explicit Hyperparameter Search Space
LEARNING_RATES = [1e-4, 2e-4, 1e-3, 2e-3]
UNCLAMP_FAPE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]
SIDE_CHAIN_FAPE_OPTIONS = [True, False]
CLAMP_FAPE_THRESHOLDS = [10.0, 20.0, 30.0, 40.0]
ENABLE_DISTOGRAM_LOSS_OPTIONS = [True, False]
ENABLE_LDDT_LOSS_OPTIONS = [True, False]


def set_fixed_seeds() -> None:
    """
    Fixes random seeds across PyTorch and Numpy environments.

    This utility ensures that every trial in the hyperparameter sweep starts with 
    the exact same weight initialization and data shuffling sequence. In the context 
    of the global project, this is critical during memorization tests; if the network 
    starts from different random states, it becomes impossible to determine if a better 
    result was caused by a superior hyperparameter combination or just a lucky initialization.

    Returns:
        None
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial: optuna.Trial) -> float:
    """
    The core evaluation function executed by Optuna for every hyperparameter trial.

    This function bridges the Optuna optimization engine with the project's native 
    training loop. It samples a specific combination of hyperparameters from the 
    defined search space, dynamically overrides the values in `cuda_configuration.yaml`, 
    and executes `main.main()`. After training completes, it cleans up and renames 
    the experiment outputs (Weights, Tensorboard, metrics CSV) to isolate the trial 
    data before returning the final distance metric to the Optuna engine for evaluation.

    Args:
        trial (optuna.Trial): The current optimization trial object provided by Optuna, 
            used to sample hyperparameter values dynamically.

    Returns:
        float: The final mean distance delta achieved at the end of the training loop.
            Optuna will attempt to minimize this metric over subsequent trials.
    """
    # Enforce identical starting weights & shuffling for every trial
    set_fixed_seeds()

    # 1. Define Search Space using explicit categorical choices
    learning_rate = trial.suggest_categorical("learning_rate", LEARNING_RATES)
    unclamp_fape_ratio = trial.suggest_categorical("unclamp_fape_ratio", UNCLAMP_FAPE_RATIOS)
    enable_side_chain_fape_loss = trial.suggest_categorical("enable_side_chain_fape_loss", SIDE_CHAIN_FAPE_OPTIONS)
    clamp_fape_threshold = trial.suggest_categorical("clamp_fape_threshold", CLAMP_FAPE_THRESHOLDS)
    enable_distogram_loss = trial.suggest_categorical("enable_distogram_loss", ENABLE_DISTOGRAM_LOSS_OPTIONS)
    enable_lddt_loss = trial.suggest_categorical("enable_lddt_loss", ENABLE_LDDT_LOSS_OPTIONS)

    # 2. Modify Configuration
    # We edit the actual configuration file that main.py will read from.
    config_path = PROJECT_ROOT / "configurations" / "cuda_configuration.yaml"

    with open(config_path, "r") as file_handle:
        config = yaml.safe_load(file_handle)

    # Inject overrides (using the exact yaml keys defined in cuda_configuration.yaml)
    config["ExperimentConfiguration"]["learning_rate"] = learning_rate
    config["ExperimentConfiguration"]["number_iterations"] = TOTAL_ITERATIONS
    config["ExperimentConfiguration"]["information_dump"] = INFORMATION_DUMP_FREQUENCY

    config["GlobalConfiguration"]["unclamp_fape_ratio"] = unclamp_fape_ratio
    config["GlobalConfiguration"]["enable_side_chain_fape_loss"] = enable_side_chain_fape_loss
    config["GlobalConfiguration"]["clamp_fape_threshold"] = clamp_fape_threshold
    config["GlobalConfiguration"]["enable_distogram_loss"] = enable_distogram_loss
    config["GlobalConfiguration"]["enable_lddt_loss"] = enable_lddt_loss

    with open(config_path, "w") as file_handle:
        yaml.dump(config, file_handle)

    # 3. Run Training Loop
    # We call main() which will now return the final mean_distance_delta
    final_distance = main.main()

    # 4. Archive the full experiment folder to isolate runs
    # The active experiment folder is constructed by the parent and name in the yaml
    experiment_parent = Path(config["ExperimentConfiguration"]["experiment_parent_folder"])
    experiment_name = config["ExperimentConfiguration"]["experiment_name"]
    experiment_folder = experiment_parent / experiment_name

    # We rename the entire experiment folder by appending the trial number
    if experiment_folder.exists():
        # Dump the hyperparameter choices to a JSON file inside the folder before renaming
        hyper_params_file = experiment_folder / "hyper_parameter_configuration.json"
        with open(hyper_params_file, "w") as f:
            json.dump(trial.params, f, indent=4)

        shutil.move(
            str(experiment_folder),
            str(experiment_parent / f"{experiment_name}_trial_{trial.number}")
        )

    # Optuna will attempt to minimize this return value
    return final_distance


if __name__ == "__main__":
    # Calculate total combinations in the search space
    total_combinations = (
            len(LEARNING_RATES) *
            len(UNCLAMP_FAPE_RATIOS) *
            len(SIDE_CHAIN_FAPE_OPTIONS) *
            len(CLAMP_FAPE_THRESHOLDS) *
            len(ENABLE_DISTOGRAM_LOSS_OPTIONS) *
            len(ENABLE_LDDT_LOSS_OPTIONS)
    )

    print(f"Creating or loading Optuna study: {STUDY_NAME}")
    print(f"Total parameter combinations space: {total_combinations}")

    # Optuna uses TPESampler by default for Bayesian Optimization
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DATABASE_URL,
        direction="minimize",
        load_if_exists=True
    )

    print(f"Starting Hyperparameter Sweep.")
    print(f"Run 'optuna-dashboard {DATABASE_URL}' in another terminal to monitor progress interactively!")

    # Optimize using Bayesian optimization (TPE). We cap the trials at total_combinations.
    # TPE will intelligently adapt its search instead of blindly iterating.
    study.optimize(objective, n_trials=total_combinations)

    print("\nSweep Completed! Best trial:")
    best_trial = study.best_trial
    print(f"  Lowest Mean Distance Delta: {best_trial.value}")
    print("  Optimal Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
