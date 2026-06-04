from dataclasses import dataclass
import numpy as np

from utilities.constants import xxx_to_index


@dataclass(frozen=True)
class Atom:
    """Human-readable representation of an atom in the structure data."""
    name: str
    element: int
    charge: int
    experimental_coordinates: np.ndarray  # Choose which one to use for training
    ideal_coordinates: np.ndarray  # Choose which one to use for training (emanate
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class Residue:
    """Human-readable representation of a residue in the structure data."""
    name: str
    amino_acid_index: int
    residue_index: int  # Index in of the residue in the chain so first residue is 1, second 2 and so on..
    atom_start_index: int
    atom_count: int
    center_atom_index: int
    pseudo_carbon_beta_atom_index: int
    is_standard: bool
    is_present: bool


@dataclass(frozen=True)
class Chain:
    """Human-readable representation of a chain in the structure data."""
    name: str
    molecule_type_id: int
    entity_id: int
    instance_index: int
    chain_index: int
    atom_start_index: int
    atom_count: int
    residue_start_index: int
    residue_count: int
    cyclic_period: int


def decode_atom_name(encoded_name: np.ndarray) -> str:
    """
    Decodes Boltz integer-encoded atom names back to strings.
    Boltz encodes characters by subtracting 32 from their ASCII value.
    """
    # Inverse: chr(value + 32)
    chars = [chr(int(c) + 32) for c in encoded_name if c != 0]
    return "".join(chars).strip()


def humanize_atom(raw_atom_data) -> Atom:
    """
    Translates a single row of raw structured NumPy atom data into an Atom dataclass.
    """
    return Atom(
        name=decode_atom_name(raw_atom_data[0]),
        element=int(raw_atom_data[1]),
        charge=int(raw_atom_data[2]),
        experimental_coordinates=np.array(raw_atom_data[3]),
        ideal_coordinates=np.array(raw_atom_data[4]),
        is_present=bool(raw_atom_data[5]),
        chirality=int(raw_atom_data[6])
    )


def humanize_residue(raw_residue_data) -> Residue:
    """
    Translates a single row of raw structured NumPy residue data into a Residue dataclass.
    """
    return Residue(
        name=str(raw_residue_data[0]),
        amino_acid_index=int(raw_residue_data[1]),
        # todo TO BE VERIFIED :: Boltz has a different naming convention than what was chosen in constant.py
        #  therefore we will change the type_id here by using what is xxx_to_index dictionary on int(raw_residue_data[0]) instead of int(raw_residue_data[1])
        residue_index=int(raw_residue_data[2]),
        atom_start_index=int(raw_residue_data[3]),
        atom_count=int(raw_residue_data[4]),
        center_atom_index=int(raw_residue_data[5]),
        pseudo_carbon_beta_atom_index=int(raw_residue_data[6]),
        is_standard=bool(raw_residue_data[7]),
        is_present=bool(raw_residue_data[8])
    )


def humanize_chain(raw_chain_data) -> Chain:
    """
    Translates a single row of raw structured NumPy chain data into a Chain dataclass.
    Handles cases where 'cyclic_period' might be missing.
    """
    return Chain(
        name=str(raw_chain_data[0]),
        molecule_type_id=int(raw_chain_data[1]),
        entity_id=int(raw_chain_data[2]),
        instance_index=int(raw_chain_data[3]),
        chain_index=int(raw_chain_data[4]),
        atom_start_index=int(raw_chain_data[5]),
        atom_count=int(raw_chain_data[6]),
        residue_start_index=int(raw_chain_data[7]),
        residue_count=int(raw_chain_data[8]),
        # Handle cases where cyclic_period is not present (older datasets)
        cyclic_period=int(raw_chain_data[9]) if len(raw_chain_data) > 9 else 0
    )


def humanize_npz_structure_data(key, data):
    """
    Legacy wrapper for humanizing data. 
    Maintains compatibility but leverages the new dataclasses internally.
    """
    if key == 'atoms' or key == 'atom':
        if isinstance(data, np.ndarray) and data.ndim > 0:
            return [humanize_atom(row) for row in data]
        return humanize_atom(data)

    if key == 'residues' or key == 'residue':
        if isinstance(data, np.ndarray) and data.ndim > 0:
            return [humanize_residue(row) for row in data]
        return humanize_residue(data)

    if key == 'chains' or key == 'chain':
        if isinstance(data, np.ndarray) and data.ndim > 0:
            return [humanize_chain(row) for row in data]
        return humanize_chain(data)

    # Generic fallback for keys we have not analyzed yet
    humanized_dict = {}
    for i, item in enumerate(data):
        humanized_dict[f"unknown_{key}_field_{i}"] = item

    return humanized_dict
