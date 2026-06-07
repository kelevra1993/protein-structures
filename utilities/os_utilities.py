import json
import numpy as np
import io
from utilities.constants import atom_types
import modelcif
import modelcif.model
import modelcif.dumper
import yaml
from pathlib import Path
from typing import Dict, Any


def read_npz_file(path: str):
    return np.load(path, allow_pickle=True)


def read_json(path: str):
    with open(path, "r") as file:
        json_data = json.loads(file.read())

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


def to_modelcif(atom_positions, atom_mask, sequence):
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
