from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from utilities.os_utilities import read_json

from dataclasses import dataclass
from utilities.constants import (xxx_to_index, index_to_xxx, atom_to_index, atom_frame_indices,
                                 rigid_group_atom_position_map, chi_angles_mask,
                                 chi_angles_frame_centers, chi_dihedral_dictionary,
                                 rigid_group_atom_positions)
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

    def __init__(self, npz_path: str, record_path: Optional[str] = None,
                 device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        """
        Initializes the Structure object by loading and parsing an NPZ file.

        Args:
            npz_path (str): Path to the .npz file containing structured protein data.
            record_path (str, optional): Path to the corresponding JSON record file for metadata.
            device (torch.device, optional): The target device.
            dtype (torch.dtype, optional): The target data type.
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
        self.device = device
        self.dtype = dtype

        if record_path is not None:
            record_data = read_json(record_path)
            structure_meta = record_data.get("structure", {})
            self.resolution = structure_meta.get("resolution")
            self.method = structure_meta.get("method")

        # Compute ground truth tensors once
        (self.ground_truth_global_positions,
         self.ground_truth_local_positions,
         self.ground_truth_frames, self.ground_truth_angles) = self.compute_ground_truth_data(
            device=self.device, dtype=self.dtype)

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
        Translates raw residue data from the NPZ file into a list of Residue dataclasses.

        This method performs identity lookups for canonical amino acids using the
        project's global residue dictionary to assign standard indices.

        Args:
            raw_residues (np.ndarray): Structured NumPy array containing residue metadata.

        Returns:
            list[Residue]: A list of populated Residue objects.
        """
        return [Residue(name=str(row[0]),
                        amino_acid_index=xxx_to_index.get(str(row[0])),
                        residue_index=int(row[2]),
                        atom_start_index=int(row[3]),
                        atom_count=int(row[4]),
                        center_atom_index=int(row[5]),
                        pseudo_carbon_beta_atom_index=int(row[6]),
                        is_standard=bool(row[7]),
                        is_present=bool(row[8])) for row in raw_residues]

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

        # Special Case: Some datasets (like OpenFold NPZs) systematically miss the OD1 atom for ASN.
        # If it is missing but we have ND2, we can geometrically reconstruct it.
        if residue_object.name == "ASN":
            self._repair_asn_residue(atom_dictionary, device, dtype)

        return atom_dictionary

    @staticmethod
    def _repair_asn_residue(atom_dictionary: Dict[str, Dict],
                            device: torch.device, dtype: torch.dtype) -> None:
        """
        Reconstructs the missing OD1 atom for an ASN residue using the verified Planar Projection method.

        Asparagine amide groups are trigonal planar. We can use the coordinates of
        CB, CG, and ND2 to find the plane and the correct position for OD1.

        Our Assumptions :
        * The distance from the center (CG) to OD1 is fixed (the bond length).
        * The angle between the CG-ND2 bond and the CG-OD1 bond is fixed (~122.6 degrees).

        Args:
            atom_dictionary (Dict[str, Dict]): The residue's atom dictionary.
            device (torch.device): Computation device.
            dtype (torch.dtype): Computation data type.
        """
        # If OD1 is already present or required atoms are missing, do nothing
        if "OD1" in atom_dictionary:
            return

        expected_atoms = ["CB", "CG", "ND2"]
        if not all(name in atom_dictionary for name in expected_atoms):
            print(f"Impossible To Repair Asparagine : ")
            print(f" - Found {list(set(list(atom_dictionary.keys())).intersection(set(expected_atoms)))}")
            print(f" - Expecting {expected_atoms}")
            exit()

        carbon_beta_global_position = atom_dictionary["CB"]["global_position"]
        carbon_gamma_global_position = atom_dictionary["CG"]["global_position"]
        nitrogen_amide_global_position = atom_dictionary["ND2"]["global_position"]

        # 1. Define a temporary planar frame centered at CG
        # ex: vector from CB to CG
        # ey: vector from CG to ND2 (defines the amide plane)
        ex_vector = carbon_gamma_global_position - carbon_beta_global_position
        ey_vector = nitrogen_amide_global_position - carbon_gamma_global_position

        # Create the transform. This puts ND2 in the XY plane with y > 0.
        planar_transformation = create_4x4_transform_matrix(ex=ex_vector, ey=ey_vector,
                                                            translation_vector=carbon_gamma_global_position)

        inverse_planar_transformation = invert_4x4_transform_matrix(planar_transformation)

        # 2. Determine the orientation of ND2 in this local plane
        nitrogen_amide_local_position = apply_transformation_on_vector(
            transformation_matrix=inverse_planar_transformation,
            vector=nitrogen_amide_global_position)

        angle_nitrogen_amide = torch.atan2(input=nitrogen_amide_local_position[1],
                                           other=nitrogen_amide_local_position[0])

        # 3. Calculate OD1 position using ideal trigonal planar geometry
        # Verified offset: Angle_OD1 = Angle_ND2 - 122.6 degrees
        angle_offset = torch.deg2rad(torch.tensor(122.6, device=device, dtype=dtype))
        angle_oxygen_amide = angle_nitrogen_amide - angle_offset

        # Ideal CG-OD1 bond length is ~1.2338 Angstroms
        # In reality we couldn't have directly used
        # "OD1" ideal position (0.633, 1.059, 0.000) since there could be a sign problem
        oxygen_amide_local_position = torch.tensor(data=[
            1.2338 * torch.cos(angle_oxygen_amide),  # x coordinate
            1.2338 * torch.sin(angle_oxygen_amide),  # y coordinate
            0.0],  # z coordinate
            device=device, dtype=dtype)

        # 4. Project the reconstructed OD1 back to the global coordinate system
        oxygen_amide_global_position = apply_transformation_on_vector(transformation_matrix=planar_transformation,
                                                                      vector=oxygen_amide_local_position)

        # 5. Inject the reconstructed atom into the dictionary
        atom_dictionary["OD1"] = {
            "global_position": oxygen_amide_global_position,
            "frame_index": 5,  # ASN OD1 is in Chi2 (Frame 5)
            "local_position": oxygen_amide_global_position.clone(),
            "current_frame_used": None,
            "atom_index": atom_to_index["OD1"]}

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
    def _apply_transform(atom_dictionary: Dict[str, Dict],
                         frame_index: int,
                         inverse_transformation_matrix: torch.Tensor,
                         apply_hierarchical: bool = True) -> None:
        """
        Helper function to apply an inverse transformation to atoms belonging to a specific frame.

        Args:
            atom_dictionary (Dict[str, Dict]): Dictionary containing residue atom data.
            frame_index (int): The current frame index being processed.
            inverse_transformation_matrix (torch.Tensor): The inverse transform to apply.
            apply_hierarchical (bool): If True, applies to all downstream frames (>= frame_index).
                                     If False, applies ONLY to the current frame (== frame_index).
        """
        for atom_data in atom_dictionary.values():
            # Determine which atoms to transform based on the hierarchical flag
            if apply_hierarchical:
                should_transform = atom_data["frame_index"] >= frame_index
            else:
                should_transform = atom_data["frame_index"] == frame_index

            # Ensure the atom is present (global position not zero)
            if should_transform and sum(atom_data["global_position"]) != 0.0:
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
         - ey : vector C_i -> CA_i
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

    def _compute_chi_frames(self, residue_object: Residue, atom_dictionary: Dict[str, Dict],
                            residue_frames: torch.Tensor, residue_angles: torch.Tensor,
                            device: torch.device, dtype: torch.dtype) -> None:
        """
        Computes the Chi frames (Frames 4-7) for a residue side-chain.
        Details:
         - Frame 4 (Chi1): ex=CA->SC0, ey=CA->N, translation=SC0
         - Frame 5-7 (Chi2-4): ex=previous_origin->current_origin, ey=(-1,0,0), translation=current_origin
        Args:
            residue_object (Residue): The residue object.
            atom_dictionary (Dict[str, Dict]): Dictionary of the residue's atoms.
            residue_frames (torch.Tensor): Tensor to store computed frames (8, 4, 4).
            residue_angles (torch.Tensor): Tensor to store computed angles (7, 2).
            device (torch.device): Computation device.
            dtype (torch.dtype): Computation data type.
        """
        residue_name = residue_object.name
        amino_acid_index = residue_object.amino_acid_index
        canonical_residue_name = index_to_xxx[amino_acid_index]

        # Iterate through the 4 possible chi angles
        for chi_index in range(4):
            # Check if this chi angle exists for the current residue
            if chi_angles_mask[amino_acid_index][chi_index] == 0:
                residue_angles[chi_index + 3] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
                continue

            frame_index = chi_index + 4

            # Basis vectors and origin depend on whether it's Chi1 or Chi2-4
            if chi_index == 0:
                # Chi1 (Frame 4)
                sidechain_0_name = chi_angles_frame_centers[canonical_residue_name][0]
                sidechain_1_name = chi_dihedral_dictionary[canonical_residue_name]["atom_1"]

                # Check if all required atoms for Chi1 are present
                required_atoms = ["N", "CA", sidechain_0_name, sidechain_1_name]
                if not all(name in atom_dictionary for name in required_atoms):
                    residue_angles[chi_index + 3] = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
                    continue

                sidechain_0_local_position = atom_dictionary[sidechain_0_name]["local_position"]
                carbon_alpha_local_position = atom_dictionary["CA"]["local_position"]
                nitrogen_local_position = atom_dictionary["N"]["local_position"]

                ex = sidechain_0_local_position - carbon_alpha_local_position
                ey = nitrogen_local_position - carbon_alpha_local_position
                translation_vector = sidechain_0_local_position

                # Dihedral: N, CA, SC0, SC1
                point_1 = atom_dictionary["N"]["global_position"]
                point_2 = atom_dictionary["CA"]["global_position"]
                point_3 = atom_dictionary[sidechain_0_name]["global_position"]
                point_4 = atom_dictionary[sidechain_1_name]["global_position"]
            else:
                # Chi2, Chi3, Chi4 (Frames 5, 6, 7)
                previous_sidechain_name = chi_angles_frame_centers[canonical_residue_name][chi_index - 1]
                current_sidechain_name = chi_angles_frame_centers[canonical_residue_name][chi_index]
                dihedral_atom_4_name = chi_dihedral_dictionary[canonical_residue_name][f"atom_{chi_index + 1}"]

                # Determine point_1 based on chi level
                if chi_index == 1:
                    point_1_name = "CA"
                else:
                    # For Chi3 (idx 2) -> SC0; For Chi4 (idx 3) -> SC1
                    point_1_name = chi_angles_frame_centers[canonical_residue_name][chi_index - 2]

                point_1 = atom_dictionary[point_1_name]["global_position"]
                current_sidechain_local_position = atom_dictionary[current_sidechain_name]["local_position"]

                # ex is vector from parent origin to current origin. parent origin is (0,0,0)
                ex = current_sidechain_local_position
                ey = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype)
                translation_vector = current_sidechain_local_position

                # Dihedral: SC_{n-2}, SC_{n-1}, SC_n, SC_{n+1}
                point_2 = atom_dictionary[previous_sidechain_name]["global_position"]
                point_3 = atom_dictionary[current_sidechain_name]["global_position"]
                point_4 = atom_dictionary[dihedral_atom_4_name]["global_position"]

            # Compute dihedral angle
            chi_angle = compute_dihedral_angle(point_1=point_1, point_2=point_2, point_3=point_3, point_4=point_4)

            # Build transformation
            chi_base_transformation = create_4x4_transform_matrix(ex=ex, ey=ey, translation_vector=translation_vector)
            rotation_chi = make_transformation_matrix_around_ex(phi=chi_angle)
            chi_transformation_matrix = torch.matmul(chi_base_transformation, rotation_chi)

            # Store results
            residue_angles[chi_index + 3] = chi_angle
            residue_frames[frame_index] = chi_transformation_matrix

            # Absolutely necessary to apply hierarchical update of coordinates
            inverse_chi_transformation_matrix = invert_4x4_transform_matrix(chi_transformation_matrix)
            self._apply_transform(atom_dictionary=atom_dictionary,
                                  frame_index=frame_index,
                                  inverse_transformation_matrix=inverse_chi_transformation_matrix,
                                  apply_hierarchical=True)

    def compute_ground_truth_data(self,
                                  device: torch.device, dtype: torch.dtype,
                                  debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Orchestrates the computation of ground truth positions, frames, and angles for all residues.

        Args:
        device (torch.device): Computation device.
        dtype (torch.dtype): Computation data type.

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ground_truth_global_positions: (number_residues, 37, 3)
            - ground_truth_local_positions: (number_residues, 37, 3)
            - ground_truth_frames: (number_residues, 8, 4, 4)
            - ground_truth_angles: (number_residues, 7, 2)
        """

        # - ground_truth_positions: (number_residues, 37, 3)
        ground_truth_local_positions = torch.zeros((self.number_residues, 37, 3), device=device, dtype=dtype)
        ground_truth_global_positions = torch.zeros((self.number_residues, 37, 3), device=device, dtype=dtype)

        # - ground_truth_frames: (number_residues, 8, 4, 4)
        ground_truth_frames = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(
            self.number_residues, 8, 1, 1)

        # - ground_truth_angles: (number_residues, 7, 2)
        ground_truth_angles = torch.zeros((self.number_residues, 7, 2), device=device, dtype=dtype)

        for residue_index, residue_object in enumerate(self.residues):
            # 1. Initialize atom dictionary
            atom_dictionary = self._get_residue_atom_dictionary(residue_object, device, dtype)

            # 2. Compute Backbone Frame (Frame 0)
            backbone_frame, inverse_backbone_frame = self._compute_backbone_frame(atom_dictionary)
            ground_truth_frames[residue_index, 0] = backbone_frame

            # 3. Compute Omega Frame (Frame 1)
            omega_frame, omega_angle = self._compute_omega_frame(
                residue_index=residue_index,
                atom_dictionary=atom_dictionary,
                inverse_backbone_transformation_matrix=inverse_backbone_frame,
                device=device, dtype=dtype)
            ground_truth_frames[residue_index, 1] = omega_frame
            ground_truth_angles[residue_index, 0] = omega_angle

            # 4. Compute Phi Frame (Frame 2)
            phi_frame, phi_angle = self._compute_phi_frame(residue_index=residue_index,
                                                           atom_dictionary=atom_dictionary,
                                                           device=device, dtype=dtype)
            ground_truth_frames[residue_index, 2] = phi_frame
            ground_truth_angles[residue_index, 1] = phi_angle

            # No Hierarchical update for Phi (Frame 2) since no atoms in the 37-set belong to it
            # Keeping the comment for consistency with other steps and as a note.

            # 5. Compute Psi Frame (Frame 3)
            psi_frame, psi_angle = self._compute_psi_frame(atom_dictionary=atom_dictionary, device=device, dtype=dtype)
            ground_truth_frames[residue_index, 3] = psi_frame
            ground_truth_angles[residue_index, 2] = psi_angle

            # Hierarchical update for Psi (Frame 3) - Only for atoms of frame 3
            # (Psi does not lead to any downstream frames in the 37-atom set)
            inverse_psi_frame = invert_4x4_transform_matrix(psi_frame)
            self._apply_transform(atom_dictionary=atom_dictionary,
                                  frame_index=3,
                                  inverse_transformation_matrix=inverse_psi_frame,
                                  apply_hierarchical=False)

            # 6. Compute Chi Frames (Frames 4-7)
            self._compute_chi_frames(residue_object=residue_object,
                                     atom_dictionary=atom_dictionary,
                                     residue_frames=ground_truth_frames[residue_index],
                                     residue_angles=ground_truth_angles[residue_index],
                                     device=device, dtype=dtype)

            # 7. Map local positions to ground_truth_positions
            for atom_data in atom_dictionary.values():
                atom_index = atom_data["atom_index"]
                ground_truth_local_positions[residue_index, atom_index] = atom_data["local_position"]
                ground_truth_global_positions[residue_index, atom_index] = atom_data["global_position"]

            # Last Step For Debugging Frames
            if debug:
                self.frame_debugger(atom_dictionary=atom_dictionary, residue_name=residue_object.name,
                                    frame_to_consider=None, threshold=0.01)

        return ground_truth_global_positions, ground_truth_local_positions, ground_truth_frames, ground_truth_angles

    @staticmethod
    def frame_debugger(atom_dictionary: Dict[str, Dict],
                       residue_name: str,
                       frame_to_consider: Optional[int] = None,
                       threshold: Optional[float] = None) -> None:

        """
        Debugs the local coordinates of atoms by comparing them to constant reference positions.

        Args:
        atom_dictionary (Dict[str, Dict]): Dictionary containing residue atom data.
        residue_name (str): The name of the residue (e.g., 'ARG').
        frame_to_consider (int, optional): The specific frame index to debug.
        threshold (float, optional): The norm threshold for reporting deltas. If None, reports non-zero norms.
        """

        for atom_name, atom_data in atom_dictionary.items():
            atom_frame_index = atom_data["frame_index"]
            current_frame_used = atom_data["current_frame_used"]

            if frame_to_consider is not None and frame_to_consider != atom_frame_index:
                continue

            if current_frame_used == atom_frame_index:
                local_position = atom_data["local_position"].numpy()
                constant_position = rigid_group_atom_position_map[residue_name][atom_name].numpy()
                difference = local_position - constant_position
                difference_norm = np.linalg.norm(difference)

                # Determine if we should print based on the threshold
                should_print = False
                if threshold is None:
                    if difference_norm != 0.0:
                        should_print = True
                elif difference_norm > threshold:
                    should_print = True

                if should_print:
                    print(40 * '-')
                    print(f"Residue: {residue_name} | Atom: {atom_name} | Frame: {atom_frame_index}")
                    print(f"Computed Local: {local_position.round(4)}")
                    print(f"Constant Local: {constant_position.round(4)}")
                    print(f"Delta: {difference.round(4)} | Norm: {difference_norm.round(4)}")
                    print(40 * '-')
