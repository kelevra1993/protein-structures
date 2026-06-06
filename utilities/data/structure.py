from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from utilities.os_utilities import read_json

from dataclasses import dataclass
from utilities.constants import (xxx_to_index, atom_to_index, atom_frame_indices,
                                 rigid_group_atom_position_map)
from utilities.geometry_utilities import (create_4x4_transform_matrix,
                                          invert_4x4_transform_matrix,
                                          apply_transformation_on_vector,
                                          compute_dihedral_angle,
                                          make_transformation_matrix_around_ex)


@dataclass(frozen=True)
class Atom:
    """
    Human-readable representation of an atom in the structure data.

    Attributes:
        name (str): The string name of the atom (e.g., 'N', 'CA', 'C', 'O', 'CB').
        element (int): The atomic number of the element (e.g., 6 for Carbon, 7 for Nitrogen).
        charge (int): The formal charge of the atom.
        experimental_coordinates (np.ndarray): 3D coordinates [x, y, z] from the experimental structure (e.g., PDB/CIF).
            Note: Values may be [0, 0, 0] if the atom is not present (see is_present).
        ideal_coordinates (np.ndarray): 3D coordinates [x, y, z] of the atom in a canonical, local frame.
            Centering and orientation depend on the residue type and the generation method (e.g., RDKit).
        is_present (bool): Boolean mask indicating if the atom was experimentally resolved.
        chirality (int): Integer encoding representing the chirality type of the atom.
    """
    name: str
    element: int
    charge: int
    experimental_coordinates: np.ndarray
    ideal_coordinates: np.ndarray
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class Residue:
    """
    Human-readable representation of a residue in the structure data.

    Attributes:
        name (str): The 3-letter code for the residue (e.g., 'MET', 'ARG', 'MSE').
        amino_acid_index (int): The integer mapping to the canonical amino acid identity.
            Based on xxx_to_index. Standardizes variants like MSE to MET (index 12).
        residue_index (int): The 0-based positional index of this residue within its chain sequence.
        atom_start_index (int): The starting index of this residue's atoms in the global structure atoms list.
        atom_count (int): The total number of atoms associated with this residue.
        center_atom_index (int): The global atom index used to define the origin of the local coordinate frame.
            Typically points to the C-alpha (CA) atom for proteins.
        pseudo_carbon_beta_atom_index (int): The global atom index used for distance calculations (distograms).
            Points to Carbon-beta (CB) for 19 amino acids, and C-alpha (CA) for Glycine.
        is_standard (bool): True if the residue is one of the 20 canonical amino acids.
        is_present (bool): True if at least some part of the residue is resolved in the experimental structure.
    """
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
    """
    Human-readable representation of a chain (polymer strand or ligand) in the structure data.

    Attributes:
        name (str): The PDB identifier for the chain (e.g., 'A', 'B', 'H').
        molecule_type_id (int): Integer category of the molecule (0: Protein, 1: DNA, 2: RNA, 3: Non-Polymer).
        entity_id (int): Unique identifier for each distinct sequence in the complex.
            Multiple chains with the same sequence will share the same entity_id.
        instance_index (int): 0-based index identifying which copy of a unique entity this chain represents.
        chain_index (int): Absolute sequential index of this chain within the structure's full list of chains.
        atom_start_index (int): The starting index of this chain's atoms in the global structure atoms list.
        atom_count (int): The total number of atoms in this chain.
        residue_start_index (int): The starting index of this chain's residues in the global structure residues list.
        residue_count (int): The total number of residues in this chain.
    """
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

    This class parses raw structured NumPy arrays into lists of descriptive dataclass instances
    for atoms, residues, and chains, while applying custom standardization logic.

    Attributes:
        boltz_filter (np.ndarray): Boolean mask indicating which chains passed quality filters during preprocessing.
        atoms (list[Atom]): List of all atoms in the complex.
        residues (list[Residue]): List of all residues in the complex.
        chains (list[Chain]): List of all chains (protein strands, ligands, etc.) in the complex.
        number_chains (int): Total number of chains (derived from NPZ).
        number_residues (int): Total number of residues (derived from NPZ).
        resolution (float, optional): Experimental resolution of the structure, loaded from JSON record.
        method (str, optional): Experimental method used to determine the structure, loaded from JSON record.
    """

    def __init__(self, npz_path: str, record_path: Optional[str] = None):
        """
        Initializes the Structure object by loading and parsing an NPZ file.

        Args:
            npz_path (str): Path to the .npz file containing structured protein data.
            record_path (str, optional): Path to the corresponding JSON record file for metadata.
        """
        data = np.load(npz_path, allow_pickle=True)

        self.boltz_filter = data['mask']
        self.atoms = self._get_atoms(data['atoms'])
        self.residues = self._get_residues(raw_residues=data['residues'])
        self.chains = self._get_chains(raw_chains=data['chains'])

        self.number_atoms = len(self.atoms)
        self.number_chains = len(self.chains)
        self.number_residues = len(self.residues)
        self.resolution = None
        self.method = None

        if record_path is not None:
            record_data = read_json(record_path)
            structure_meta = record_data.get("structure", {})
            self.resolution = structure_meta.get("resolution")
            self.method = structure_meta.get("method")

    @staticmethod
    def decode_atom_name(encoded_name: np.ndarray) -> str:
        """
        Decodes Boltz integer-encoded atom names back to strings.

        Boltz encodes characters by subtracting 32 from their ASCII value.
        Zeros are treated as padding.
        """
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
        """
        Translates all raw residue rows into Residue dataclasses with custom indexing.

        Standardizes residue names (e.g., MSE -> MET) and performs identity lookups
        using the project's canonical amino acid dictionary.
        """
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

    def _get_residue_atom_dictionary(self, residue_object: Residue,
                                     device: torch.device, dtype: torch.dtype) -> Dict[str, Dict]:
        """
        Maps a residue's atoms to the canonical 37-atom representation and initializes the atom dictionary.

        Args:
            residue_object (Residue): The residue object to process.
            device (torch.device): The device to use for tensor creation.
            dtype (torch.dtype): The data type to use for tensors.

        Returns:
            Dict[str, Dict]: A dictionary containing atom data for iterative frame transformations.
        """
        atom_dictionary = {}

        # Go through all the atoms of the provided residue
        for index in range(residue_object.atom_count):

            atom_object = self.atoms[residue_object.atom_start_index + index]
            atom_name = atom_object.name

            if atom_name in atom_to_index:
                atom_index = atom_to_index[atom_name]
                global_position = torch.tensor(atom_object.experimental_coordinates, device=device, dtype=dtype)

                # Get frame index for the specific amino acid type and atom type
                frame_index = int(atom_frame_indices[residue_object.amino_acid_index, atom_index])

                # local_position : for iteratively updating the postion through local frames
                # frame_index : target frame for the atom
                # current_frame_used : tracking the current local frame
                # both frame_index and current_frame_used help us debug the code
                atom_dictionary[atom_name] = {"global_position": global_position,
                                              "frame_index": frame_index,
                                              "local_position": global_position.clone(),
                                              "current_frame_used": None,
                                              "atom_index": atom_index}
        return atom_dictionary

    @staticmethod
    def _compute_backbone_frame(atom_dictionary: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the backbone frame (Frame 0) and transforms all residue atoms into this local frame.
        Details :
         - ex : vector CA->C
         - ey : vector CA->N
         - translation : CA position
        Args:
            atom_dictionary (Dict[str, Any]): Dictionary containing residue atom coordinates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The backbone transformation matrix and its inverse.
        """
        carbon_alpha_global_position = atom_dictionary["CA"]["global_position"]
        carbon_global_position = atom_dictionary["C"]["global_position"]
        nitrogen_global_position = atom_dictionary["N"]["global_position"]

        # Get base vectors
        ex = carbon_global_position - carbon_alpha_global_position
        ey = nitrogen_global_position - carbon_alpha_global_position

        backbone_transformation_matrix = create_4x4_transform_matrix(
            ex=ex, ey=ey, translation_vector=carbon_alpha_global_position)

        inverse_backbone_transformation_matrix = invert_4x4_transform_matrix(backbone_transformation_matrix)

        # Express all atoms in the backbone frame (Frame 0)
        for atom_data in atom_dictionary.values():
            atom_data["local_position"] = apply_transformation_on_vector(
                transformation_matrix=inverse_backbone_transformation_matrix, vector=atom_data["local_position"])
            atom_data["current_frame_used"] = 0

        return backbone_transformation_matrix, inverse_backbone_transformation_matrix

    @staticmethod
    def _apply_hierarchical_transform(atom_dictionary: Dict[str, Dict],
                                      frame_index: int,
                                      inverse_transformation_matrix: torch.Tensor) -> None:
        """
        Helper function to apply an inverse transformation to all atoms at or downstream of a frame.

        Args:
            atom_dictionary (Dict[str, Dict]): Dictionary containing residue atom data.
            frame_index (int): The current frame index being processed.
            inverse_transformation_matrix (torch.Tensor): The inverse transform to apply.
        """
        for atom_data in atom_dictionary.values():
            # Apply to atoms in this frame or any downstream frames (hierarchical logic)
            # and ensure the atom is present (global position not zero)
            if atom_data["frame_index"] >= frame_index and sum(atom_data["global_position"]) != 0.0:
                atom_data["local_position"] = apply_transformation_on_vector(
                    transformation_matrix=inverse_transformation_matrix,
                    vector=atom_data["local_position"])

                atom_data["current_frame_used"] = frame_index

    def _compute_omega_frame(self, residue_index: int, atom_dictionary: Dict[str, Dict],
                             inverse_backbone_transformation_matrix: torch.Tensor,
                             device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Omega frame (Frame 1) for the peptide bond between residue i and i+1.
        Details:
         - ex : vector C_i -> N_{i+1}
         - ey : vector CA_i -> C_i
         - translation : C_i position (expressed in the backbone frame)
        Args:
            residue_index (int): Index of the current residue.
            atom_dictionary (Dict[str, Dict]): Dictionary of the current residue's atoms.
            inverse_backbone_transformation_matrix (torch.Tensor): Inverse of current residue's Frame 0.
            device (torch.device): Computation device.
            dtype (torch.dtype): Computation data type.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Omega transformation matrix and the (cos, sin) angle.
        """
        # Current residue global positions for dihedral and local basis
        carbon_alpha_global_position = atom_dictionary["CA"]["global_position"]
        carbon_global_position = atom_dictionary["C"]["global_position"]

        # Check for next residue to define peptide bond
        if residue_index < self.number_residues - 1:
            next_residue_object = self.residues[residue_index + 1]
            next_nitrogen_global_position = None
            next_carbon_alpha_global_position = None

            # Find required atoms in the next residue
            for i in range(next_residue_object.atom_count):
                atom_object = self.atoms[next_residue_object.atom_start_index + i]
                if atom_object.name == "N":
                    next_nitrogen_global_position = torch.tensor(atom_object.experimental_coordinates,
                                                                 device=device, dtype=dtype)
                elif atom_object.name == "CA":
                    next_carbon_alpha_global_position = torch.tensor(atom_object.experimental_coordinates,
                                                                     device=device, dtype=dtype)

            if next_nitrogen_global_position is not None and next_carbon_alpha_global_position is not None:
                # Actual Omega dihedral angle: CA_i, C_i, N_{i+1}, CA_{i+1}
                omega_angle = compute_dihedral_angle(point_1=carbon_alpha_global_position,
                                                     point_2=carbon_global_position,
                                                     point_3=next_nitrogen_global_position,
                                                     point_4=next_carbon_alpha_global_position)

                # Basis vectors expressed in the Backbone Frame (Frame 0)
                local_carbon = apply_transformation_on_vector(
                    transformation_matrix=inverse_backbone_transformation_matrix, vector=carbon_global_position)
                local_carbon_alpha = apply_transformation_on_vector(
                    transformation_matrix=inverse_backbone_transformation_matrix, vector=carbon_alpha_global_position)
                local_next_nitrogen = apply_transformation_on_vector(
                    transformation_matrix=inverse_backbone_transformation_matrix, vector=next_nitrogen_global_position)

                # Get base vectors
                ex = local_next_nitrogen - local_carbon
                ey = local_carbon_alpha - local_carbon

                omega_base_transformation = create_4x4_transform_matrix(ex=ex, ey=ey, translation_vector=local_carbon)

                # Rotate by the omega angle around ex
                rotation_omega = make_transformation_matrix_around_ex(phi=omega_angle)
                omega_transformation_matrix = torch.matmul(omega_base_transformation, rotation_omega)

                return omega_transformation_matrix, omega_angle

        # C-terminus or missing next atoms fallback: identity translation to C
        omega_angle = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        local_carbon = apply_transformation_on_vector(transformation_matrix=inverse_backbone_transformation_matrix,
                                                      vector=carbon_global_position)

        omega_transformation_matrix = torch.eye(4, device=device, dtype=dtype)
        omega_transformation_matrix[:3, 3] = local_carbon

        return omega_transformation_matrix, omega_angle

    def _compute_phi_frame(self, residue_index: int, atom_dictionary: Dict[str, Dict],
                           device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Phi frame (Frame 2).
        Details:
         - ex : vector CA -> N
         - ey : vector CA -> C
         - translation : N position (expressed in the current local frame)
        Args:
            residue_index (int): Index of the current residue.
            atom_dictionary (Dict[str, Dict]): Dictionary of the current residue's atoms.
            device (torch.device): Computation device.
            dtype (torch.dtype): Computation data type.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Phi transformation matrix and the (cos, sin) angle.
        """
        # Global positions for dihedral calculation
        nitrogen_global_position = atom_dictionary["N"]["global_position"]
        carbon_alpha_global_position = atom_dictionary["CA"]["global_position"]
        carbon_global_position = atom_dictionary["C"]["global_position"]

        # Local positions for basis vector creation
        nitrogen_local_position = atom_dictionary["N"]["local_position"]
        carbon_alpha_local_position = atom_dictionary["CA"]["local_position"]
        carbon_local_position = atom_dictionary["C"]["local_position"]

        # Dihedral angle: C_{i-1}, N_i, CA_i, C_i
        phi_angle = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        if residue_index > 0:
            previous_residue_object = self.residues[residue_index - 1]
            previous_carbon_global_position = None
            for i in range(previous_residue_object.atom_count):
                atom_object = self.atoms[previous_residue_object.atom_start_index + i]
                if atom_object.name == "C":
                    previous_carbon_global_position = torch.tensor(atom_object.experimental_coordinates,
                                                                   device=device, dtype=dtype)
                    break

            if previous_carbon_global_position is not None:
                phi_angle = compute_dihedral_angle(point_1=previous_carbon_global_position,
                                                   point_2=nitrogen_global_position,
                                                   point_3=carbon_alpha_global_position,
                                                   point_4=carbon_global_position)

        # Get base vectors
        ex = nitrogen_local_position - carbon_alpha_local_position
        ey = carbon_local_position - carbon_alpha_local_position

        phi_base_transformation = create_4x4_transform_matrix(ex=ex, ey=ey, translation_vector=nitrogen_local_position)

        # Rotate by the phi angle around ex
        rotation_phi = make_transformation_matrix_around_ex(phi=phi_angle)
        phi_transformation_matrix = torch.matmul(phi_base_transformation, rotation_phi)

        return phi_transformation_matrix, phi_angle

    @staticmethod
    def _compute_psi_frame(atom_dictionary: Dict[str, Dict],
                           device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Psi frame (Frame 3).
        Details:
         - ex : vector CA -> C
         - ey : vector CA -> N
         - translation : C position (expressed in the current local frame)
        Args:
            atom_dictionary (Dict[str, Dict]): Dictionary of the current residue's atoms.
            device (torch.device): Computation device.
            dtype (torch.dtype): Computation data type.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Psi transformation matrix and the (cos, sin) angle.
        """
        # Global positions for dihedral calculation
        nitrogen_global_position = atom_dictionary["N"]["global_position"]
        carbon_alpha_global_position = atom_dictionary["CA"]["global_position"]
        carbon_global_position = atom_dictionary["C"]["global_position"]
        oxygen_global_position = atom_dictionary["O"]["global_position"]

        # Local positions for basis vector creation
        nitrogen_local_position = atom_dictionary["N"]["local_position"]
        carbon_alpha_local_position = atom_dictionary["CA"]["local_position"]
        carbon_local_position = atom_dictionary["C"]["local_position"]

        # Actual Psi dihedral angle: N_i, CA_i, C_i, O_i
        psi_angle = compute_dihedral_angle(point_1=nitrogen_global_position,
                                           point_2=carbon_alpha_global_position,
                                           point_3=carbon_global_position,
                                           point_4=oxygen_global_position)

        # Get base vectors
        ex = carbon_local_position - carbon_alpha_local_position
        ey = nitrogen_local_position - carbon_alpha_local_position

        psi_base_transformation = create_4x4_transform_matrix(ex=ex, ey=ey, translation_vector=carbon_local_position)

        # Rotate by the psi angle around ex
        rotation_psi = make_transformation_matrix_around_ex(phi=psi_angle)
        psi_transformation_matrix = torch.matmul(psi_base_transformation, rotation_psi)

        return psi_transformation_matrix, psi_angle
