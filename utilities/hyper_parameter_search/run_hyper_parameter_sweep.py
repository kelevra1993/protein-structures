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
# Upgraded to v2 to allow the expanded LEARNING_RATES while keeping the same database file
STUDY_NAME = "overfit_sweep_v2"

# Explicit Hyperparameter Search Space
LEARNING_RATES = [1e-4, 2e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
UNCLAMP_FAPE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SIDE_CHAIN_FAPE_OPTIONS = [True, False]
CLAMP_FAPE_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
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


# Globally store configs parsed from existing folders to avoid duplicate runs
EVALUATED_CONFIGS_FROM_FOLDERS = []

def load_previous_configs():
    """Scans the experiment directory for past runs and records their hyperparameters and final metrics."""
    global EVALUATED_CONFIGS_FROM_FOLDERS
    EVALUATED_CONFIGS_FROM_FOLDERS.clear()
    config_path = PROJECT_ROOT / "configurations" / "cuda_configuration.yaml"
    if not config_path.exists():
        return
        
    with open(config_path, "r") as file_handle:
        config = yaml.safe_load(file_handle)
        
    experiment_parent = Path(config["ExperimentConfiguration"]["experiment_parent_folder"])
    experiment_name = config["ExperimentConfiguration"]["experiment_name"]
    
    if experiment_parent.exists():
        for d in experiment_parent.iterdir():
            if d.is_dir() and f"{experiment_name}_trial_" in d.name:
                json_path = d / "hyper_parameter_configuration.json"
                csv_path = d / "metrics_evolution.csv"
                if json_path.exists() and csv_path.exists():
                    try:
                        with open(json_path, "r") as f:
                            params = json.load(f)
                            
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        if "Mean_Distance_Delta" in df.columns:
                            final_dist = float(df["Mean_Distance_Delta"].iloc[-1])
                        elif "mean_distance_delta" in df.columns:
                            final_dist = float(df["mean_distance_delta"].iloc[-1])
                        else:
                            continue
                            
                        EVALUATED_CONFIGS_FROM_FOLDERS.append({
                            "params": params,
                            "value": final_dist
                        })
                    except Exception:
                        pass
    print(f"Loaded {len(EVALUATED_CONFIGS_FROM_FOLDERS)} previously completed configurations from disk.")


def objective(trial: optuna.Trial) -> float:
    """
    The core evaluation function executed by Optuna for every hyperparameter trial.
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

    # 2. Check for Duplicates / Fast-Forward
    # Check against folder-based history
    for past_run in EVALUATED_CONFIGS_FROM_FOLDERS:
        if trial.params == past_run["params"]:
            print(f"Fast-forwarding previously completed trial. Returning saved value: {past_run['value']}")
            return past_run["value"]
            
    # Check against Optuna study database history (for actual duplicates during new search)
    for past_trial in trial.study.trials:
        if past_trial.state == optuna.trial.TrialState.COMPLETE and past_trial.params == trial.params and past_trial.number != trial.number:
            raise optuna.exceptions.TrialPruned("Duplicate parameter combination (already in database).")
    # 3. Modify Configuration
    config_path = PROJECT_ROOT / "configurations" / "cuda_configuration.yaml"

    with open(config_path, "r") as file_handle:
        config = yaml.safe_load(file_handle)

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

    # 4. Run Training Loop
    final_distance = main.main()

    # 5. Archive the folder and calculate the NEXT trial number properly
    experiment_parent = Path(config["ExperimentConfiguration"]["experiment_parent_folder"])
    experiment_name = config["ExperimentConfiguration"]["experiment_name"]
    experiment_folder = experiment_parent / experiment_name

    if experiment_folder.exists():
        # Find the highest existing trial suffix to avoid overwriting old folders
        max_idx = -1
        for d in experiment_parent.iterdir():
            if d.is_dir() and f"{experiment_name}_trial_" in d.name:
                try:
                    idx = int(d.name.split("_trial_")[-1])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        
        next_trial_num = max_idx + 1

        # Dump the hyperparameter choices
        hyper_params_file = experiment_folder / "hyper_parameter_configuration.json"
        with open(hyper_params_file, "w") as f:
            json.dump(trial.params, f, indent=4)

        # Move with the dynamically calculated next trial number
        shutil.move(
            str(experiment_folder),
            str(experiment_parent / f"{experiment_name}_trial_{next_trial_num}")
        )
        
        # Add to our local cache so future trials in this run don't duplicate it
        EVALUATED_CONFIGS_FROM_FOLDERS.append({
            "params": trial.params,
            "value": final_distance
        })

    return final_distance


if __name__ == "__main__":
    # Scan disk for existing trials before starting the study
    load_previous_configs()
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

    # Using a tuned TPESampler for less aggressive, broader exploration early on.
    # n_startup_trials=50 acts as a random sampler for the first 50 runs to map the space.
    # multivariate=True helps the Bayesian model look at combinations of parameters jointly.
    tuned_sampler = optuna.samplers.TPESampler(
        n_startup_trials=50,
        multivariate=True,
        seed=42
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DATABASE_URL,
        direction="minimize",
        sampler=tuned_sampler,
        load_if_exists=True
    )

    print(f"Starting Hyperparameter Sweep.")
    print(f"Run 'optuna-dashboard {DATABASE_URL}' in another terminal to monitor progress interactively!")

    # Fast-forward old runs into the new study if it's empty
    if len(study.trials) == 0:
        print("Injecting past runs into the new study to preserve history...")
        for past_run in EVALUATED_CONFIGS_FROM_FOLDERS:
            # We enqueue the parameters. Optuna will immediately sample them.
            # Our objective function will intercept them and return the saved value instantly.
            study.enqueue_trial(past_run["params"])

    # Optimize using Bayesian optimization (TPE) capped at the total possible combinations.
    # The pruning logic will ensure we never waste time on duplicated trials.
    study.optimize(objective, n_trials=total_combinations)

    print("\nSweep Completed! Best trial:")
    best_trial = study.best_trial
    print(f"  Lowest Mean Distance Delta: {best_trial.value}")
    print("  Optimal Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
