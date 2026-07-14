import torch
import unittest
import numpy as np
from pathlib import Path
from utilities.data.structure import Structure, Residue, Atom
from utilities.constants import (atom_to_index, rigid_group_atom_position_map,
                                 index_to_atom, atom_frame_indices,
                                 peptide_carbon_nitrogen_length_base,
                                 carbon_alpha_carbon_nitrogen_angle_base,
                                 carbon_nitrogen_carbon_alpha_angle_base,
                                 next_nitrogen_elevation_angle_base)
from utilities.geometry_utilities import (apply_transformation_on_vector,
                                          reconstruct_next_nitrogen_from_scalers,
                                          reconstruct_next_carbon_alpha_from_scalers)

current_directory = Path(__file__).parent
reference_directory = current_directory / "reference_values"


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.npz_path = reference_directory / "structures" / "P90561.npz"
        self.record_path = reference_directory / "records" / "P90561.json"
        self.device = torch.device("cpu")
        self.dtype = torch.float32

        self.structure = Structure(
            npz_path=str(self.npz_path),
            record_path=str(self.record_path),
            device=self.device,
            dtype=self.dtype
        )

    def test_initialization(self):
        """Verify that the structure object is initialized correctly."""
        self.assertEqual(self.structure.number_residues, 354)
        self.assertEqual(len(self.structure.residues), 354)
        self.assertTrue(len(self.structure.atoms) > 0)
        self.assertEqual(self.structure.resolution, 0.0)
        self.assertEqual(self.structure.method, None)

    def test_ground_truth_shapes(self):
        """Check the shapes of the computed ground truth tensors."""
        self.assertEqual(self.structure.ground_truth_global_positions.shape, (354, 37, 3))
        self.assertEqual(self.structure.ground_truth_local_positions.shape, (354, 37, 3))
        self.assertEqual(self.structure.ground_truth_frames.shape, (354, 8, 4, 4))
        self.assertEqual(self.structure.ground_truth_angles.shape, (354, 7, 2))

    def test_backbone_frames_sanity(self):
        """Sanity check: Carbon Alpha should be at (0,0,0) in its own frame (Frame 0)."""
        carbon_alpha_index = atom_to_index["CA"]

        # Verify for the first 10 residues
        for residue_index in range(10):
            carbon_alpha_local_position = self.structure.ground_truth_local_positions[residue_index, carbon_alpha_index]
            torch.testing.assert_close(carbon_alpha_local_position,
                                       torch.zeros(3, device=self.device, dtype=self.dtype),
                                       atol=1e-5, rtol=1e-5)

    def test_recomputation(self):
        """Verify that recomputing ground truth yields same results."""
        (recomputed_global_positions,
         recomputed_local_positions,
         recomputed_frames,
         recomputed_angles,
         recomputed_peptide_linker_scalers) = self.structure.compute_ground_truth_data(
            device=self.device, dtype=self.dtype)

        torch.testing.assert_close(recomputed_global_positions, self.structure.ground_truth_global_positions)
        torch.testing.assert_close(recomputed_local_positions, self.structure.ground_truth_local_positions)
        torch.testing.assert_close(recomputed_frames, self.structure.ground_truth_frames)
        torch.testing.assert_close(recomputed_angles, self.structure.ground_truth_angles)

    def test_local_coordinate_precision(self):
        """
        Verify that all computed local coordinates match the canonical 
        reference positions within a strict tolerance (delta < 0.01).
        """
        computed_local_positions = self.structure.ground_truth_local_positions

        for residue_index, residue_object in enumerate(self.structure.residues):
            if not residue_object.is_standard or not residue_object.is_present:
                continue

            residue_name = residue_object.name

            # Map canonical atom positions for this residue type
            canonical_map = rigid_group_atom_position_map[residue_name]

            for atom_name, canonical_position in canonical_map.items():
                atom_idx = atom_to_index[atom_name]
                computed_position = computed_local_positions[residue_index, atom_idx]

                # We use assert_close with absolute tolerance (atol) of 0.01
                # Note: Canonical positions are tensors on CPU
                torch.testing.assert_close(
                    computed_position,
                    canonical_position.to(device=self.device, dtype=self.dtype),
                    atol=0.01, rtol=0.01,
                    msg=f"Precision failure at Residue {residue_index} ({residue_name}), Atom {atom_name}"
                )

    def test_hierarchical_frame_reconstruction(self):
        """
        Comprehensive Sanity Check: Verify that global coordinates can be 
        reconstructed by chaining the stored frames (Backbone -> Local).
        
        Formula: Global_Pos = (Global_Backbone_Frame * Local_Group_Frame) * Canonical_Pos
        """
        frames = self.structure.ground_truth_frames
        global_positions = self.structure.ground_truth_global_positions

        for residue_index, residue_object in enumerate(self.structure.residues):
            if not residue_object.is_standard or not residue_object.is_present:
                continue

            residue_name = residue_object.name
            residue_frames = frames[residue_index]

            # 1. Compute Global Frames for this residue
            # Frames 1-4 (Omega, Phi, Psi, Chi1) are relative to Backbone (Frame 0)
            # Frames 5-7 (Chi2-4) are relative to their immediate parent in the side chain
            global_residue_frames = torch.zeros_like(residue_frames)

            # Frame 0 is already global
            global_residue_frames[0] = residue_frames[0]

            # Frames 1, 2, 3, 4 are local to Backbone (0)
            for i in [1, 2, 3, 4]:
                global_residue_frames[i] = torch.matmul(global_residue_frames[0], residue_frames[i])

            # Frame 5 is local to Chi1 (4)
            global_residue_frames[5] = torch.matmul(global_residue_frames[4], residue_frames[5])
            # Frame 6 is local to Chi2 (5)
            global_residue_frames[6] = torch.matmul(global_residue_frames[5], residue_frames[6])
            # Frame 7 is local to Chi3 (6)
            global_residue_frames[7] = torch.matmul(global_residue_frames[6], residue_frames[7])

            # 2. Reconstruct and Verify
            canonical_map = rigid_group_atom_position_map[residue_name]
            for atom_name, canonical_position in canonical_map.items():
                atom_index = atom_to_index[atom_name]
                frame_index = int(atom_frame_indices[residue_object.amino_acid_index, atom_index])

                # Reconstruct global coordinate
                reconstructed_global_position = apply_transformation_on_vector(
                    transformation_matrix=global_residue_frames[frame_index],
                    vector=canonical_position.to(device=self.device, dtype=self.dtype)
                )

                actual_global_position = global_positions[residue_index, atom_index]

                # Skip if atom was not experimentally resolved
                if torch.sum(torch.abs(actual_global_position)) == 0:
                    continue

                torch.testing.assert_close(
                    reconstructed_global_position,
                    actual_global_position,
                    atol=0.01, rtol=0.01,
                    msg=f"Hierarchical reconstruction failure at Res {residue_index} ({residue_name}), "
                        f"Atom {atom_name} in Frame {frame_index}"
                )

    def test_peptide_linker_reconstruction_error(self):
        """Verify that the predicted peptide linker scalers perfectly reconstruct the next residue's N and CA."""
        # Loop through all residues except the last one (and avoid the second-to-last if it lacks a next peptide bond)
        # ground_truth_peptide_linker_scalers is shape (number_residues - 1, 4)
        for residue_index in range(self.structure.number_residues - 1):
            current_residue = self.structure.residues[residue_index]
            next_residue = self.structure.residues[residue_index + 1]

            current_atom_dictionary = self.structure._get_residue_atom_dictionary(current_residue,
                                                                                  self.structure.device,
                                                                                  self.structure.dtype)
            next_atom_dictionary = self.structure._get_residue_atom_dictionary(next_residue, self.structure.device,
                                                                               self.structure.dtype)

            current_carbon_alpha_global_position = current_atom_dictionary["CA"]["global_position"]
            current_carbon_global_position = current_atom_dictionary["C"]["global_position"]
            current_oxygen_global_position = current_atom_dictionary["O"]["global_position"]
            next_nitrogen_global_position = next_atom_dictionary["N"]["global_position"]
            next_carbon_alpha_global_position = next_atom_dictionary["CA"]["global_position"]

            # Skip if any of the critical backbone atoms are not resolved (positions are exactly 0)
            if (torch.sum(torch.abs(current_carbon_alpha_global_position)) == 0 or torch.sum(
                    torch.abs(current_carbon_global_position)) == 0 or
                    torch.sum(torch.abs(current_oxygen_global_position)) == 0 or torch.sum(
                        torch.abs(next_nitrogen_global_position)) == 0 or
                    torch.sum(torch.abs(next_carbon_alpha_global_position)) == 0):
                continue

            # Skip if residues are not connected (distance > 5 Angstroms)
            if torch.linalg.norm(next_carbon_alpha_global_position - current_carbon_alpha_global_position) > 5.0:
                continue

            # Skip if get_statistics does not recognize a peptide bond here
            if self.structure.get_statistics(residue_index, compute_next_residue_statistics=True)[
                "peptide_bond"] is None:
                continue

            peptide_linker_scalers = self.structure.ground_truth_peptide_linker_scalers[residue_index]

            # 1. Reconstruct Nitrogen
            reconstructed_next_nitrogen_global_position = reconstruct_next_nitrogen_from_scalers(
                carbon_alpha=current_carbon_alpha_global_position,
                carbon=current_carbon_global_position,
                oxygen=current_oxygen_global_position,
                peptide_carbon_nitrogen_length=peptide_linker_scalers[2] * peptide_carbon_nitrogen_length_base,
                carbon_alpha_carbon_nitrogen_angle=peptide_linker_scalers[1] * carbon_alpha_carbon_nitrogen_angle_base,
                next_nitrogen_elevation_angle=peptide_linker_scalers[0] * next_nitrogen_elevation_angle_base
            )

            nitrogen_reconstruction_error = torch.linalg.norm(
                reconstructed_next_nitrogen_global_position - next_nitrogen_global_position).item()
            self.assertLess(nitrogen_reconstruction_error, 1e-3,
                            f"Nitrogen reconstruction error too high: "
                            f"{nitrogen_reconstruction_error} at residue {residue_index}")

            # 2. Reconstruct Carbon Alpha
            omega_dihedral_angle = self.structure.ground_truth_angles[residue_index, 0]  # shape: (2,) for omega
            # Convert sin, cos back to angle for the reconstruction function
            omega_degrees = torch.rad2deg(torch.atan2(omega_dihedral_angle[1], omega_dihedral_angle[0]))

            nitrogen_carbon_alpha_length = torch.linalg.norm(
                next_carbon_alpha_global_position - next_nitrogen_global_position)

            reconstructed_next_carbon_alpha_global_position = reconstruct_next_carbon_alpha_from_scalers(
                carbon_alpha=current_carbon_alpha_global_position,
                carbon=current_carbon_global_position,
                next_nitrogen=reconstructed_next_nitrogen_global_position,
                nitrogen_carbon_alpha_length=nitrogen_carbon_alpha_length,
                carbon_nitrogen_carbon_alpha_angle=peptide_linker_scalers[3] * carbon_nitrogen_carbon_alpha_angle_base,
                omega_dihedral_angle=omega_degrees
            )

            carbon_alpha_reconstruction_error = torch.linalg.norm(
                reconstructed_next_carbon_alpha_global_position - next_carbon_alpha_global_position).item()
            self.assertLess(carbon_alpha_reconstruction_error, 1e-3,
                            f"Carbon Alpha reconstruction error too high: {carbon_alpha_reconstruction_error} at residue {residue_index}")


if __name__ == "__main__":
    unittest.main()
