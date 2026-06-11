import os
import json
from json import JSONDecodeError

import torch
import numpy as np
import io
from utilities.constants import atom_types
import modelcif
import modelcif.model
import modelcif.dumper
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


def read_npz_file(path: str) -> Any:
    """
    Reads an NPZ file from the specified path.

    This utility function is used to load data stored in NumPy's NPZ format,
    typically pre-processed Multiple Sequence Alignment (MSA) features or
    predicted protein structures.

    Args:
        path (str): The file path to the .npz file.

    Returns:
        np.lib.npyio.NpzFile: A dictionary-like object containing the loaded data.
    """
    return np.load(path, allow_pickle=True)


def read_json(path: str) -> dict[str, Any] | list[Any]:
    """
    Reads and parses a JSON file.

    This function is used throughout the project to load records, manifests,
    and other metadata stored in JSON format.

    Args:
        path (str): The file path to the .json file.

    Returns:
        Dict[str, Any] | List[Any]: The parsed JSON data.
    """
    json_data = None
    try:
        with open(path, "r") as file:
            json_data = json.loads(file.read())
        return json_data
    except JSONDecodeError:
        return json_data


def load_configuration(configuration_path: str | Path) -> Dict[str, Any]:
    """
    Parses a YAML configuration file into a dictionary.

    Args:
        configuration_path (str | Path): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(configuration_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            configuration = yaml.safe_load(file)
            return configuration if configuration is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {path}:\n{e}")


def load_experiment_configuration(configuration_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a single YAML configuration file and splits it into experiment
    and model configurations.

    This function extracts the 'ExperimentConfiguration' section, processes
    its paths and types, and returns it alongside the rest of the configuration
    parameters.

    Args:
        configuration_path (str | Path): Path to the consolidated YAML file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - experiment_configuration: The processed experiment settings.
            - model_configuration: The remaining configuration (all other keys).

    Raises:
        KeyError: If 'ExperimentConfiguration' is missing from the file.
    """
    configuration = load_configuration(configuration_path)
    if "ExperimentConfiguration" not in configuration:
        raise KeyError(f"Key 'ExperimentConfiguration' not found in {configuration_path}")

    experiment_configuration = configuration.pop("ExperimentConfiguration")
    model_configuration = configuration

    # Convert paths
    path_keys = [
        "experiment_parent_folder", "data_folder", "configuration_path",
        "train_split_file", "validation_split_file", "test_split_file"
    ]
    for key in path_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = Path(experiment_configuration[key])

    # Convert numerics
    int_keys = ["information_dump", "weight_saving_iterations", "number_iterations", "precomputed_samples"]
    for key in int_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = int(float(experiment_configuration[key]))

    if "learning_rate" in experiment_configuration:
        experiment_configuration["learning_rate"] = float(experiment_configuration["learning_rate"])

    # Convert dtype
    if "dtype" in experiment_configuration:
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        experiment_configuration["dtype"] = dtype_map.get(experiment_configuration["dtype"], torch.float32)

    # Set the project root
    if "experiment_parent_folder" in experiment_configuration and "experiment_name" in experiment_configuration:
        experiment_configuration["project_root"] = (
                experiment_configuration["experiment_parent_folder"] / experiment_configuration["experiment_name"])

    return experiment_configuration, model_configuration


def to_modelcif(atom_positions: torch.Tensor, atom_mask: torch.Tensor, sequence: str | list[str]) -> str:
    """
    Converts predicted atom positions to ModelCIF format.

    ModelCIF is an extension of the PDBx/mmCIF format used for macromolecular
    structural models. This function prepares the predicted structure for
    export and visualization.

    Args:
        atom_positions (torch.Tensor): Tensor containing the 3D coordinates
            for all atoms in the protein.
            Shape: (number_residues, number_atom_types, 3)
        atom_mask (torch.Tensor): Binary mask indicating which atoms are
            present/valid.
            Shape: (number_residues, number_atom_types)
        sequence (str | List[str]): The amino acid sequence of the protein.

    Returns:
        str: A string containing the ModelCIF data in mmCIF format.
    """
    atom_positions = atom_positions.to('cpu').numpy()
    atom_mask = atom_mask.to('cpu').numpy()
    n = atom_positions.shape[0]
    system = modelcif.System(title='AlphaFold prediction')
    entity = modelcif.Entity(sequence, description='Model subunit')
    asym_unit = modelcif.AsymUnit(entity, details='Model subunit A', id='A')
    modeled_assembly = modelcif.Assembly([asym_unit], name='Modeled assembly')

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            for i in range(n):
                for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
                    if not mask:
                        continue
                    element = atom_name[0]
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit,
                        type_symbol=element,
                        seq_id=i + 1,
                        atom_id=atom_name,
                        x=pos[0], y=pos[1], z=pos[2],
                        het=False,
                        occupancy=1.00
                    )

    model = _MyModel(assembly=modeled_assembly, name='Model')
    model_group = modelcif.model.ModelGroup([model], name='All models')
    system.model_groups.append(model_group)
    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])

    return fh.getvalue()


def print_blue(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold blue color.

    This utility is used throughout the project to highlight informational
    messages, status updates, and progress indicators during model training
    or data processing.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[94m" + "\033[1m" + output + "\033[0m")
        print("\033[94m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold green color.

    This utility is typically used to indicate successful operations, such
    as completed training iterations, saved model weights, or successful
    data extraction.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[32m" + "\033[1m" + output + "\033[0m")
        print("\033[32m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold yellow color.

    This utility is used for warnings or important notices that require
    user attention but are not necessarily critical failures (e.g., missing
    optional configuration fields).

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[93m" + "\033[1m" + output + "\033[0m")
        print("\033[93m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold red color.

    This utility is reserved for error messages, critical failures, and
    exceptions that might halt the execution of the model or data pipeline.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
        print("\033[91m" + "\033[1m" + output + "\033[0m")
        print("\033[91m" + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output: str, add_separators: bool = False) -> None:
    """
    Prints a string to the console in bold font.

    This utility is used for general emphasis in console output, often for
    headers or key parameters in the experiment logs.

    Args:
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal
            separators for better visibility.
    """
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print("\033[1m" + str(length * "-") + "\033[0m")
        print("\033[1m" + output + "\033[0m")
        print("\033[1m" + str(length * "-") + "\033[0m")
    else:
        print("\033[1m" + output + "\033[0m")


def print_dictionary(dictionary: Dict[str, Any], indent: int = 4) -> None:
    """
    Prints a dictionary to the console in a formatted JSON-like style.

    This utility is used to display configuration parameters, experiment
    summaries, or manifest data in a readable format during execution.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be printed.
        indent (int): The number of spaces to use for indentation.
    """
    print(json.dumps(dictionary, indent=indent))


def globalise_path(absolute_parent_path: Path, target_path: Path) -> Path:
    """
    Ensures a path is absolute, resolving it relative to a parent if needed.

    In the project's multi-platform environment (DGX, Mac, etc.), file paths
    in configurations may be specified as relative to the experiment root.
    This function ensures that the system can always locate these files by
    falling back to the `absolute_parent_path` if the `target_path` does not
    exist independently.

    Args:
        absolute_parent_path (Path): The base directory to use for resolution.
        target_path (Path): The path to be globalised/resolved.

    Returns:
        Path: The resolved absolute path.
    """
    absolute_path = target_path

    if not absolute_path.exists():
        print_blue(f"{absolute_path} Not Found... Globalising It")
        absolute_path = absolute_parent_path / absolute_path

        if not absolute_path.exists():
            print_red(f"{absolute_path} Still does not exist")
            return target_path

    return absolute_path
