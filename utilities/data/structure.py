import numpy as np

from dataclasses import dataclass
from utilities.constants import xxx_to_index


@dataclass(frozen=True)
class Atom:
    """Human-readable representation of an atom in the structure data."""
    name: str
    element: int
    charge: int
    experimental_coordinates: np.ndarray
    ideal_coordinates: np.ndarray
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class Residue:
    """Human-readable representation of a residue in the structure data."""
    name: str
    amino_acid_index: int
    residue_index: int
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


class Structure:
    """
    Represents the full structure of a protein complex loaded from an NPZ file.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.boltz_filter = data['mask']
        self.atoms = self._get_atoms(data['atoms'])
        self.residues = self._get_residues(raw_residues=data['residues'])
        self.chains = self._get_chains(raw_chains=data['chains'])

    @staticmethod
    def decode_atom_name(encoded_name: np.ndarray) -> str:
        """Decodes Boltz integer-encoded atom names back to strings."""
        chars = [chr(int(c) + 32) for c in encoded_name if c != 0]
        return "".join(chars).strip()

    def _get_atoms(self, raw_atoms: np.ndarray) -> list[Atom]:
        """Translates all raw atom rows into Atom dataclasses."""
        return [Atom(name=self.decode_atom_name(row[0]),
                     element=int(row[1]),
                     charge=int(row[2]),
                     experimental_coordinates=np.array(row[3]),
                     ideal_coordinates=np.array(row[4]),
                     is_present=bool(row[5]),
                     chirality=int(row[6])) for row in raw_atoms]

    @staticmethod
    def _get_residues(raw_residues: np.ndarray) -> list[Residue]:
        """Translates all raw residue rows into Residue dataclasses with custom indexing."""
        parsed_residues = []
        for row in raw_residues:
            name = str(row[0])
            # TODO Try to see if we have this in our database (MSE)
            # Mapping logic: MSE -> MET, everything else non-standard -> 20 (UNK)
            parsed_name = "MET" if name == "MSE" else name
            amino_acid_index = xxx_to_index.get(parsed_name, 20)

            parsed_residues.append(Residue(name=name,
                                           amino_acid_index=amino_acid_index,
                                           residue_index=int(row[2]),
                                           atom_start_index=int(row[3]),
                                           atom_count=int(row[4]),
                                           center_atom_index=int(row[5]),
                                           pseudo_carbon_beta_atom_index=int(row[6]),
                                           is_standard=bool(row[7]),
                                           is_present=bool(row[8])))
        return parsed_residues

    @staticmethod
    def _get_chains(raw_chains: np.ndarray) -> list[Chain]:
        """Translates all raw chain rows into Chain dataclasses."""
        return [Chain(name=str(row[0]),
                      molecule_type_id=int(row[1]),
                      entity_id=int(row[2]),
                      instance_index=int(row[3]),
                      chain_index=int(row[4]),
                      atom_start_index=int(row[5]),
                      atom_count=int(row[6]),
                      residue_start_index=int(row[7]),
                      residue_count=int(row[8])) for row in raw_chains]
