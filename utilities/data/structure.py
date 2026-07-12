from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from utilities.os_utilities import read_json, print_dictionary

from dataclasses import dataclass
from utilities.constants import (xxx_to_index, index_to_xxx, atom_to_index, atom_frame_indices,
                                 rigid_group_atom_position_map, chi_angles_mask,
                                 chi_angles_frame_centers, chi_dihedral_dictionary,
                                 rigid_group_atom_positions)
from utilities.geometry_utilities import (create_4x4_transform_matrix,
                                          invert_4x4_transform_matrix,
                                          apply_transformation_on_vector,
                                          compute_angle,
                                          compute_dihedral_angle,
                                          make_transformation_matrix_around_ex, adjust_vector_angle, check_coplanarity,
                                          reconstruct_next_nitrogen_from_scalers,
                                          reconstruct_next_carbon_alpha_from_scalers)

from utilities.tensor_utilities import print_tensor_list


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

        # Compute Structure Statistics To Get Peptide Linker Scalers
        self.statistics = self.compute_all_backbone_statistics()

        # Compute ground truth tensors once
        (self.ground_truth_global_positions,
         self.ground_truth_local_positions,
         self.ground_truth_frames, self.ground_truth_angles,
         self.ground_truth_peptide_linker_scalers) = self.compute_ground_truth_data(
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
                        amino_acid_index=xxx_to_index.get(str(row[0]), 20),
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

                # local_position : for iteratively updating the position through local frames
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
                                  debug: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        # todo docstring to be updated
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
        ground_truth_peptide_linker_scalers = torch.ones((self.number_residues - 1, 4), device=device, dtype=dtype)

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

            # We will create a variable called peptide_linker_scalers
            # This variable contains three elements which are actually three scaler values with values between 0.5 and 1.5
            # peptide_linker_scalers = [scaler_angle_OCN_from_120, scaler_peptide_bond_CN_from_1.32, scaler_angle_CNCA_from_120]
            # these are actually what i will later make my model predict for the peptide linkage prediction to make training easier
            if residue_index < self.number_residues - 2:
                # 8. Fetch raw geometric values for the peptide bond connecting to the next residue
                carbon_alpha_carbon_nitrogen_angle = self.statistics["bond_angles"]["peptide_CA_C_N"][residue_index]
                peptide_carbon_nitrogen_length = self.statistics["bond_lengths"]["peptide_C_N"][residue_index]
                carbon_nitrogen_carbon_alpha_angle = self.statistics["bond_angles"]["peptide_C_N_CA"][residue_index]

                # 1. Compute the scalers relative to idealized geometry
                # Scaler values represent the deviation from idealized peptide bond geometries
                # and are normalized to fall generally between 0.5 and 1.5  for stable network prediction.
                # scalers can be larger in the case of the elevation

                # TODO Add small comment to explain what we are doing here
                next_residue_object = self.residues[residue_index + 1]
                # Find required atoms in the next residue
                next_atom_dictionary = self._get_residue_atom_dictionary(next_residue_object,
                                                                         device=self.device, dtype=self.dtype)
                coplanarity, nitrogen_elevation = check_coplanarity(
                    point_a=atom_dictionary["CA"]["global_position"],
                    point_b=atom_dictionary["O"]["global_position"],
                    point_c=next_atom_dictionary["N"]["global_position"],
                    point_center=atom_dictionary["C"]["global_position"])

                scaler_next_nitrogen_elevation_from_5 = nitrogen_elevation / 5.0
                scaler_carbon_alpha_carbon_nitrogen_angle_from_120 = carbon_alpha_carbon_nitrogen_angle / 120.0
                scaler_peptide_carbon_nitrogen_bond_from_1_32 = peptide_carbon_nitrogen_length / 1.32
                scaler_carbon_nitrogen_carbon_alpha_angle_from_120 = carbon_nitrogen_carbon_alpha_angle / 120.0

                # 2. Assemble the predictor target list
                peptide_linker_scalers = [
                    scaler_next_nitrogen_elevation_from_5,
                    scaler_carbon_alpha_carbon_nitrogen_angle_from_120,
                    scaler_peptide_carbon_nitrogen_bond_from_1_32,
                    scaler_carbon_nitrogen_carbon_alpha_angle_from_120,
                ]

                ground_truth_peptide_linker_scalers[residue_index] = torch.tensor(peptide_linker_scalers,
                                                                                  device=self.device, dtype=self.dtype)

                # DEBUGGING: Verify the mathematical reconstruction is lossless
                if debug:
                    reconstructed_next_nitrogen = reconstruct_next_nitrogen_from_scalers(
                        carbon_alpha=atom_dictionary["CA"]["global_position"],
                        carbon=atom_dictionary["C"]["global_position"],
                        oxygen=atom_dictionary["O"]["global_position"],
                        peptide_carbon_nitrogen_length=scaler_peptide_carbon_nitrogen_bond_from_1_32 * 1.32,
                        carbon_alpha_carbon_nitrogen_angle=scaler_carbon_alpha_carbon_nitrogen_angle_from_120 * 120.0,
                        next_nitrogen_elevation_angle=scaler_next_nitrogen_elevation_from_5 * 5.0)

                    reconstructed_next_carbon_alpha = reconstruct_next_carbon_alpha_from_scalers(
                        carbon_alpha=atom_dictionary["CA"]["global_position"],
                        carbon=atom_dictionary["C"]["global_position"],
                        next_nitrogen=reconstructed_next_nitrogen,
                        nitrogen_carbon_alpha_length=self.statistics["bond_lengths"]["N_CA"][residue_index + 1],
                        carbon_nitrogen_carbon_alpha_angle=scaler_carbon_nitrogen_carbon_alpha_angle_from_120 * 120.0,
                        omega_dihedral_angle=self.statistics["dihedrals"]["omega"][residue_index])

                    true_next_nitrogen = next_atom_dictionary["N"]["global_position"]
                    n_error = torch.linalg.norm(reconstructed_next_nitrogen - true_next_nitrogen).item()

                    true_next_carbon_alpha = next_atom_dictionary["CA"]["global_position"]
                    reconstruction_error = torch.linalg.norm(
                        reconstructed_next_carbon_alpha - true_next_carbon_alpha).item()

                    print(
                        f"[{residue_object.name}] N Error: {n_error:.6f} | CA Error: {reconstruction_error:.6f} Angstroms")

            # Last Step For Debugging Frames
            if debug:
                self.frame_debugger(atom_dictionary=atom_dictionary, residue_name=residue_object.name,
                                    frame_to_consider=None, threshold=0.01)

        return (ground_truth_global_positions, ground_truth_local_positions, ground_truth_frames, ground_truth_angles,
                ground_truth_peptide_linker_scalers)

    def compute_all_backbone_statistics(self,
                                        round_decimals: Optional[int] = 4) -> Dict[str, Dict[str, list[float]]]:
        """
        Computes all internal backbone coordinates (bond lengths, bond angles, and dihedral (only omega))
        for the entire protein structure.

        In the global project context, this method provides the foundational ground-truth
        statistics necessary to debug the model when training on internal coordinates
        rather than predicting direct Cartesian translations.

        Args:
            round_decimals (Optional[int]): Number of decimal places to round values to. If None, no rounding is performed. Defaults to 4.

        Returns:
            Dict[str, Dict[str, list[float]]]: A dictionary containing dictionaries of lists.
                Each list has a length equal to `number_residues` and contains the float
                value of the statistic for that residue (or None if uncomputable).
        """
        statistics = {
            "bond_lengths": {"N_CA": [], "CA_C": [], "C_O": [], "peptide_C_N": []},
            "bond_angles": {"N_CA_C": [], "CA_C_O": [],
                            "peptide_CA_C_N": [], "peptide_C_N_CA": [], "peptide_O_C_N": []},
            "dihedrals": {"omega": []}}

        number_residues = len(self.residues)
        for residue_index in range(number_residues):

            # 1. Fetch current residue atoms explicitly
            residue_atoms = self._get_residue_atom_dictionary(self.residues[residue_index],
                                                              device=self.device, dtype=self.dtype)

            nitrogen_position = residue_atoms["N"]["global_position"]
            carbon_alpha_position = residue_atoms["CA"]["global_position"]
            carbon_position = residue_atoms["C"]["global_position"]
            oxygen_position = residue_atoms["O"]["global_position"]

            # 2. Compute intra-residue bond lengths
            nitrogen_carbon_alpha_length = torch.linalg.norm(carbon_alpha_position - nitrogen_position).item()
            carbon_alpha_carbon_length = torch.linalg.norm(carbon_position - carbon_alpha_position).item()
            carbon_oxygen_length = torch.linalg.norm(oxygen_position - carbon_position).item()

            statistics["bond_lengths"]["N_CA"].append(nitrogen_carbon_alpha_length)
            statistics["bond_lengths"]["CA_C"].append(carbon_alpha_carbon_length)
            statistics["bond_lengths"]["C_O"].append(carbon_oxygen_length)

            # 3. Compute intra-residue bond angles
            nitrogen_carbon_alpha_carbon_angle = compute_angle(point_a=nitrogen_position,
                                                               point_b=carbon_position,
                                                               center=carbon_alpha_position)[1].item()

            carbon_alpha_carbon_oxygen_angle = compute_angle(point_a=carbon_alpha_position,
                                                             point_b=oxygen_position,
                                                             center=carbon_position)[1].item()

            statistics["bond_angles"]["N_CA_C"].append(nitrogen_carbon_alpha_carbon_angle)
            statistics["bond_angles"]["CA_C_O"].append(carbon_alpha_carbon_oxygen_angle)

            # 5. Compute peptide bond properties Omega (requires next residue)
            omega_angle = None
            peptide_carbon_nitrogen_length = None
            peptide_carbon_alpha_carbon_nitrogen_angle = None
            peptide_carbon_nitrogen_carbon_alpha_angle = None
            peptide_oxygen_carbon_nitrogen_angle = None

            if residue_index < number_residues - 1:
                next_atoms = self._get_residue_atom_dictionary(self.residues[residue_index + 1],
                                                               device=self.device, dtype=self.dtype)
                next_nitrogen = next_atoms["N"]["global_position"]
                next_carbon_alpha = next_atoms["CA"]["global_position"]

                # Inter-residue lengths and angles
                peptide_carbon_nitrogen_length = torch.linalg.norm(next_nitrogen - carbon_position).item()
                peptide_carbon_alpha_carbon_nitrogen_angle = compute_angle(point_a=carbon_alpha_position,
                                                                           point_b=next_nitrogen,
                                                                           center=carbon_position)[1].item()
                peptide_carbon_nitrogen_carbon_alpha_angle = compute_angle(point_a=carbon_position,
                                                                           point_b=next_carbon_alpha,
                                                                           center=next_nitrogen)[1].item()
                peptide_oxygen_carbon_nitrogen_angle = compute_angle(point_a=oxygen_position,
                                                                     point_b=next_nitrogen,
                                                                     center=carbon_position)[1].item()

                # Omega dihedral
                dihedral_angle_result = compute_dihedral_angle(point_1=carbon_alpha_position,
                                                               point_2=carbon_position,
                                                               point_3=next_nitrogen,
                                                               point_4=next_carbon_alpha)
                omega_angle = torch.rad2deg(torch.atan2(dihedral_angle_result[1], dihedral_angle_result[0])).item()

            statistics["bond_lengths"]["peptide_C_N"].append(peptide_carbon_nitrogen_length)
            statistics["bond_angles"]["peptide_CA_C_N"].append(peptide_carbon_alpha_carbon_nitrogen_angle)
            statistics["bond_angles"]["peptide_C_N_CA"].append(peptide_carbon_nitrogen_carbon_alpha_angle)
            statistics["bond_angles"]["peptide_O_C_N"].append(peptide_oxygen_carbon_nitrogen_angle)
            statistics["dihedrals"]["omega"].append(omega_angle)

        # Rounding values for readability
        if round_decimals is not None:
            for category, metrics in statistics.items():
                for metric_name, values in metrics.items():
                    statistics[category][metric_name] = [
                        round(val, round_decimals) if val is not None else None for val in values]

        return statistics

    def get_statistics(self, residue_index: int, compute_next_residue_statistics: bool = True, debug=False) -> dict:
        """
        Retrieves the isolated structural statistics for a specific residue.

        In the global project context, this utility allows for readable, localized inspection
        of internal coordinates for debugging and reporting purposes.

        Args:
            residue_index (int): The index of the residue to inspect.
            compute_next_residue_statistics (bool, optional): Whether to append inter-residue
                measurements and the immediately adjacent (i+1) residue. Defaults to True.
            debug (bool, optional): Whether to print additional debug information. Defaults to False.

        Returns:
            dict: A formatted dictionary containing the float values for bond lengths,
                bond angles, and dihedral angles.
        """
        statistics_reference = self.statistics
        result = {"current_residue": {
            "bond_lengths": {
                "N_CA": statistics_reference["bond_lengths"]["N_CA"][residue_index],
                "CA_C": statistics_reference["bond_lengths"]["CA_C"][residue_index],
                "C_O": statistics_reference["bond_lengths"]["C_O"][residue_index]},
            "bond_angles": {
                "N_CA_C": statistics_reference["bond_angles"]["N_CA_C"][residue_index],
                "CA_C_O": statistics_reference["bond_angles"]["CA_C_O"][residue_index]}}}

        if compute_next_residue_statistics:
            if residue_index < self.number_residues - 1:
                result["peptide_bond"] = {
                    "bond_lengths": {"C_N": statistics_reference["bond_lengths"]["peptide_C_N"][residue_index]},
                    "bond_angles": {"CA_C_N": statistics_reference["bond_angles"]["peptide_CA_C_N"][residue_index],
                                    "C_N_CA": statistics_reference["bond_angles"]["peptide_C_N_CA"][residue_index],
                                    "O_C_N": statistics_reference["bond_angles"]["peptide_O_C_N"][residue_index]},
                    "omega": statistics_reference["dihedrals"]["omega"][residue_index]}

                next_residue_index = residue_index + 1
                result["next_residue"] = {
                    "bond_lengths": {"N_CA": statistics_reference["bond_lengths"]["N_CA"][next_residue_index],
                                     "CA_C": statistics_reference["bond_lengths"]["CA_C"][next_residue_index],
                                     "C_O": statistics_reference["bond_lengths"]["C_O"][next_residue_index]},
                    "bond_angles": {"N_CA_C": statistics_reference["bond_angles"]["N_CA_C"][next_residue_index],
                                    "CA_C_O": statistics_reference["bond_angles"]["CA_C_O"][next_residue_index]}}
            else:
                result["peptide_bond"] = None
                result["next_residue"] = None

        if debug:
            print_dictionary(result)

        return result

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

            if current_frame_used == atom_frame_index and atom_name in rigid_group_atom_position_map[residue_name]:
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


def visualize_frame_and_atoms(atom_positions: dict, frames, axis_scale: float = 1.0, connectivity_map: list = None):
    """
    Visualizes atoms and specific frames using Plotly for interactive 3D rendering.

    Args:
        atom_positions (dict): Dictionary formatted as {residue_index: {atom_name: [x, y, z]}}
        frames (torch.Tensor, np.ndarray, or list): 4x4 transformation matrix or a list of them.
        axis_scale (float): Scale factor for the basis vector lines.
        connectivity_map (list of tuples): List of atom name pairs to connect, e.g., [("N", "CA"), ("CA", "C")].
    """
    import plotly.graph_objects as go

    if not connectivity_map:
        connectivity_map = [("N", "CA"), ("CA", "C"), ("CA", "CB"), ("C", "O")]
    fig = go.Figure()

    # Collect coordinates for atoms (markers and text)
    x_coords, y_coords, z_coords, text_labels = [], [], [], []

    # Collect coordinates for bonds (lines)
    line_x, line_y, line_z = [], [], []

    for residue_index, atoms in atom_positions.items():
        clean_atoms = {}
        for atom_name, coords in atoms.items():
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()
            clean_atoms[atom_name] = coords

            x_coords.append(coords[0])
            y_coords.append(coords[1])
            z_coords.append(coords[2])
            text_labels.append(f"Residue {residue_index} - {atom_name}")

        if connectivity_map is not None:
            # Draw specific bonds from the map
            for atom1, atom2 in connectivity_map:
                if atom1 in clean_atoms and atom2 in clean_atoms:
                    c1, c2 = clean_atoms[atom1], clean_atoms[atom2]
                    line_x.extend([c1[0], c2[0], None])
                    line_y.extend([c1[1], c2[1], None])
                    line_z.extend([c1[2], c2[2], None])
        else:
            # Fallback: connect atoms in the order they appear
            for atom_name, coords in clean_atoms.items():
                line_x.append(coords[0])
                line_y.append(coords[1])
                line_z.append(coords[2])
            line_x.append(None)
            line_y.append(None)
            line_z.append(None)

    # Plot bonds (lines)
    if line_x:
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines',
            line=dict(color='gray', width=3),
            hoverinfo='none',
            name='Bonds'
        ))

    # Plot atoms (markers+text)
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers+text',
        marker=dict(size=4, color='black'),
        text=text_labels,
        textposition="top center",
        name='Atoms'
    ))

    # Ensure frames is a list
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
        if frames.ndim == 2:
            frames = [frames]
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 2:
            frames = [frames]
    elif not isinstance(frames, (list, tuple)):
        frames = [frames]

    for i, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # Extract origin and axes
        origin = frame[:3, 3]
        x_axis = origin + frame[:3, 0] * axis_scale
        y_axis = origin + frame[:3, 1] * axis_scale
        z_axis = origin + frame[:3, 2] * axis_scale

        frame_name_suffix = f" {i}" if len(frames) > 1 else ""

        # Plot axes
        # X-axis (Red)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]],
            mode='lines+text', line=dict(color='red', width=6),
            text=["", f"ex{frame_name_suffix}"], textposition="middle right",
            name=f'X-axis{frame_name_suffix}'
        ))

        # Y-axis (Green)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]],
            mode='lines+text', line=dict(color='green', width=6),
            text=["", f"ey{frame_name_suffix}"], textposition="top center",
            name=f'Y-axis{frame_name_suffix}'
        ))

        # Z-axis (Blue)
        fig.add_trace(go.Scatter3d(
            x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]],
            mode='lines+text', line=dict(color='blue', width=6),
            text=["", f"ez{frame_name_suffix}"], textposition="top center",
            name=f'Z-axis{frame_name_suffix}'
        ))

    # Set layout with equal aspect ratio
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Interactive Frame and Atom Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()
