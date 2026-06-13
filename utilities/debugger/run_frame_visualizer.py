"""
Script to visualize transformation frames of a protein structure from an NPZ file.
"""

import argparse
import torch
import os
from utilities.data.structure import Structure
from utilities.debugger.visualization_utilities import visualization
from pathlib import Path
from utilities.os_utilities import globalise_path

DEFAULT_NPZ = "data_examples/openfold/structures/P90561.npz"


def main():
    parser = argparse.ArgumentParser(description="Visualize protein transformation frames from an NPZ file.")
    parser.add_argument("--npz", type=str, default=DEFAULT_NPZ,
                        help=f"Path to the .npz structure file (default: {DEFAULT_NPZ}).")
    parser.add_argument("--record", type=str,
                        help="Path to the corresponding .json record file (optional).")
    parser.add_argument("--num_residues", type=int, default=50,
                        help="Number of residues to visualize.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Index of the frame to visualize (0 for backbone).")

    args = parser.parse_args()

    project_path = Path(os.getcwd()).parents[1]
    args.npz = str(globalise_path(absolute_parent_path=project_path, target_path=Path(args.npz)))

    if not os.path.exists(args.npz):
        print(f"Error: File {args.npz} not found.")
        return

    print(f"Loading structure from {args.npz}...")
    # Load structure on CPU for visualization
    structure = Structure(npz_path=args.npz, record_path=args.record, device=torch.device("cpu"))

    # Extract frames: (number_residues, 8, 4, 4)
    all_frames = structure.ground_truth_frames

    # Select the specific frame index (e.g., backbone Frame 0)
    # Shape: (number_residues, 4, 4)
    selected_frames = all_frames[:, args.frame_index]

    # Limit to the requested number of residues
    num_to_show = min(args.num_residues, selected_frames.shape[0])
    frames_to_visualize = selected_frames[:num_to_show]

    title = f"Protein Structure - Frame {args.frame_index} ({num_to_show} residues)"
    print(f"Visualizing {num_to_show} residues (Frame {args.frame_index}) using Matplotlib...")
    visualization(frames_tensor=frames_to_visualize, title=title)


if __name__ == "__main__":
    main()
