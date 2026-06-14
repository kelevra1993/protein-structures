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


def to_modelcif(atom_positions: torch.Tensor, atom_mask: torch.Tensor, sequence: str | list[str],
                description: str) -> str:
    """
    Converts predicted atom positions into a ModelCIF-formatted string.

    ModelCIF is an extension of the mmCIF format specifically designed for 
    computational structural models. This function maps raw tensor coordinates
    to the correct amino acid atoms and generates a file that can be opened 
    in visualization software like PyMOL or ChimeraX.

    Args:
        atom_positions (torch.Tensor): Predicted 3D coordinates for all atoms.
            Expected shape: (number_residues, 37, 3).
        atom_mask (torch.Tensor): Binary mask indicating valid predicted atoms.
            Expected shape: (number_residues, 37).
        sequence (str | List[str]): The amino acid sequence of the protein.

    Returns:
        str: The complete ModelCIF data as a string.
    """
    # Move to CPU and convert to NumPy for compatibility with the modelcif library
    atom_positions = atom_positions.to('cpu').detach().numpy()
    atom_mask = atom_mask.to('cpu').detach().numpy()

    number_residues = atom_positions.shape[0]

    # Initialize the ModelCIF system
    system = modelcif.System(title=description)

    # Define the protein entity and its sequence
    # If sequence is a list of characters, join it into a string
    if isinstance(sequence, list):
        sequence = "".join(sequence)

    entity = modelcif.Entity(sequence, description='Predicted Protein Chain')

    # Define the asymmetric unit (the actual chain in the model)
    asym_unit = modelcif.AsymUnit(entity, details='Model Chain A', id='A')

    # Group chains into an assembly
    modeled_assembly = modelcif.Assembly([asym_unit], name='Modeled Assembly')

    # Define a custom Model class to yield individual atoms to the dumper
    class _PredictedModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            """
            Generator that iterates through all residues and atoms, 
            yielding only those that are valid according to the mask.
            """
            for residue_index in range(number_residues):
                # We iterate through the standard 37 AlphaFold atom types
                for atom_name, atom_position, mask in zip(atom_types, atom_positions[residue_index],
                                                          atom_mask[residue_index]):
                    if not mask:
                        continue

                    # Get the element symbol (first letter of atom name, e.g., 'C' for 'CA')
                    element_symbol = atom_name[0]

                    yield modelcif.model.Atom(
                        asym_unit=asym_unit,
                        type_symbol=element_symbol,
                        seq_id=residue_index + 1,  # PDB/CIF uses 1-based indexing for residues
                        atom_id=atom_name,
                        x=atom_position[0], y=atom_position[1], z=atom_position[2],
                        het=False,  # Not a heteroatom
                        occupancy=1.00  # Standard occupancy for models
                    )

    # Create the model instance and add it to the system
    model = _PredictedModel(assembly=modeled_assembly, name=description)
    model_group = modelcif.model.ModelGroup([model], name='Atomic Models')
    system.model_groups.append(model_group)

    # Dump the system to a string buffer
    string_buffer = io.StringIO()
    modelcif.dumper.write(string_buffer, [system])

    return string_buffer.getvalue()


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
