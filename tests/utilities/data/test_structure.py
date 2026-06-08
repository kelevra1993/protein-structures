import torch
import unittest
import numpy as np
from pathlib import Path
from utilities.data.structure import Structure, Residue, Atom
from utilities.constants import (atom_to_index, rigid_group_atom_position_map, 
                                 index_to_atom, atom_frame_indices)
from utilities.geometry_utilities import apply_transformation_on_vector

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
        for i in range(10):
            carbon_alpha_local_position = self.structure.ground_truth_local_positions[i, carbon_alpha_index]
            torch.testing.assert_close(carbon_alpha_local_position, 
                                       torch.zeros(3, device=self.device, dtype=self.dtype), 
                                       atol=1e-5, rtol=1e-5)

    def test_recomputation(self):
        """Verify that recomputing ground truth yields same results."""
        (recomputed_global_positions, 
         recomputed_local_positions, 
         recomputed_frames, 
         recomputed_angles) = self.structure.compute_ground_truth_data(
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

if __name__ == "__main__":
    unittest.main()
