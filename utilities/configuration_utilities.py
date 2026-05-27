import yaml
from pathlib import Path
from typing import Dict, Any


def load_configuration(config_path: str | Path) -> Dict[str, Any]:
    """
    Parses a YAML configuration file into a dictionary.

    Args:
        config_path (str | Path): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            configuration = yaml.safe_load(file)
            return configuration if configuration is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {path}:\n{e}")
