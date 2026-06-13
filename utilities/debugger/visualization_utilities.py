"""
File containing visualization utilities for protein structures and transformations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utilities.geometry_utilities import (invert_4x4_transform_matrix,
                                          create_quaternion_from_axis_and_angle,
                                          turn_quaternion_to_3x3_matrix,
                                          assemble_4x4_transform_matrix)


def _prepare_frames(frames: torch.Tensor, center_first: bool) -> np.ndarray:
    """
    Helper function to center and convert transformation frames to NumPy.

    This utility ensures that tensors are moved to the CPU and detached from
    the computation graph before visualization. It also optionally centers all
    frames relative to the first one to simplify spatial analysis.

    Args:
        frames (torch.Tensor): Transformation matrices representing local
            coordinate systems (e.g., backbone or side-chain frames).
            Shape: (number_frames, 4, 4)
        center_first (bool): If True, multiplies all frames by the inverse
            of the first frame to center the sequence at the origin.

    Returns:
        np.ndarray: The processed frames as a NumPy array.
            Shape: (number_frames, 4, 4)
    """
    frames = frames.detach().cpu()
    if center_first:
        first_frame = frames[0]
        inv_first_frame = invert_4x4_transform_matrix(first_frame)
        frames = torch.matmul(inv_first_frame, frames)
    return frames.numpy()


def plot_transformation_frames(frames: torch.Tensor,
                               center_first: bool = True,
                               scale: float = 1.0,
                               show_trace: bool = True,
                               title: str = "Transformation Frames"):
    """
    Plots a set of transformation frames in 3D using Matplotlib.

    This function is used to visualize the predicted local coordinate systems
    of protein residues (backbone frames) or side-chain groups. It helps in
    analyzing the relative orientations and spatial distribution of these frames.

    Standard color coding:
    - Red: X-axis (direction of the peptide bond/rotation axis)
    - Green: Y-axis
    - Blue: Z-axis
    - Gray Dashed Line: Trace connecting frame origins (representing the protein backbone)

    Args:
        frames (torch.Tensor): Transformation matrices to plot.
            Shape: (number_frames, 4, 4)
        center_first (bool): If True, all frames are relative to the first frame.
        scale (float): Scaling factor for the axis vectors.
        show_trace (bool): If True, draws a line connecting the origins of the frames.
        title (str): Title for the plot.
    """
    if plt is None:
        print("Error: Matplotlib is not installed.")
        return

    frames_np = _prepare_frames(frames, center_first)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_origins = frames_np[:, :3, 3]

    # Plot Trace (C-alpha line)
    if show_trace:
        ax.plot(all_origins[:, 0], all_origins[:, 1], all_origins[:, 2],
                color='green', alpha=0.6, label='Trace', linewidth=5)

    for i, frame in enumerate(frames_np):
        origin = frame[:3, 3]
        x_axis, y_axis, z_axis = frame[:3, 0], frame[:3, 1], frame[:3, 2]

        ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2],
                  color='r', length=scale, normalize=True)
        ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2],
                  color='g', length=scale, normalize=True)
        ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2],
                  color='b', length=scale, normalize=True)

        ax.text(origin[0], origin[1], origin[2], f" {i}", color='black', fontsize=10)

    # Set equal aspect ratio logic
    min_limit = np.min(all_origins, axis=0) - scale
    max_limit = np.max(all_origins, axis=0) + scale
    max_range = (max_limit - min_limit).max() / 2.0
    mid = (max_limit + min_limit) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='green', lw=2)]
    ax.legend(custom_lines, ['X-axis', 'Y-axis', 'Z-axis', 'Trace'], loc='upper left')
    plt.show()


def generate_random_frames(number_frames: int = 20) -> torch.Tensor:
    """
    Generates a set of dummy transformation frames in a spiral pattern for testing.

    This function is a utility used to verify the correctness of the visualization
    modules and to provide a standard test case for spatial transformations. It
    creates a sequence of frames that translate along a spiral path and orient
    themselves using quaternions and assembly utilities.

    Args:
        number_frames (int): The number of frames to generate.

    Returns:
        torch.Tensor: A tensor of transformation matrices.
            Shape: (number_frames, 4, 4)
    """
    frames = []

    for i in range(number_frames):
        # 1. Translation: Spiral upwards
        angle = 0.5 * i
        x = np.cos(angle) * 2.0
        y = np.sin(angle) * 2.0
        z = i * 0.5
        translation = torch.tensor([x, y, z], dtype=torch.float32)

        # 2. Rotation: Rotate to look "forward" along the spiral path
        # Rotate around Z-axis
        phi = torch.tensor(angle, dtype=torch.float32)
        axis = torch.tensor([x, y, x], dtype=torch.float32)
        quaternion = create_quaternion_from_axis_and_angle(phi, axis)
        rotation_matrix = turn_quaternion_to_3x3_matrix(quaternion)

        # 3. Assemble
        frame = assemble_4x4_transform_matrix(rotation_matrix, translation)
        frames.append(frame)

    frames_tensor = torch.stack(frames)

    print(f"Generated {number_frames} Test Frames.")

    return frames_tensor


def visualization(frames_tensor: torch.Tensor = None, title: str = "Transformation Frames"):
    """
    High-level interface to test and display transformation frames using Matplotlib.

    This utility function acts as a wrapper for `plot_transformation_frames`. It is
    primarily used during the development and debugging phases to visually
    inspect the orientation and placement of backbone or side-chain frames
    generated by the model.

    Args:
        frames_tensor (torch.Tensor, optional): The transformation matrices to visualize.
            If None, randomly generated frames are used.
            Shape: (number_frames, 4, 4)
        title (str): The title of the plot.
    """

    if frames_tensor is None:
        frames_tensor = generate_random_frames()

    print("Launching Matplotlib visualization...")
    plot_transformation_frames(frames_tensor, center_first=False, scale=0.5, title=title)
