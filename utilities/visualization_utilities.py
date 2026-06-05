"""
File containing visualization utilities for protein structures and transformations.
"""

import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
except ImportError:
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from utilities.geometry_utilities import invert_4x4_transform_matrix


def _prepare_frames(frames: torch.Tensor, center_first: bool) -> np.ndarray:
    """Helper to center and convert frames to numpy."""
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

    Standard color coding:
    - Red: X-axis, Green: Y-axis, Blue: Z-axis
    - Gray Dashed Line: Trace connecting frame origins

    Args:
        frames (torch.Tensor): Transformation matrices. Shape: `(number_frames, 4, 4)`.
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
                color='gray', linestyle='--', alpha=0.6, label='Trace')

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

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='gray', lw=1, linestyle='--')]
    ax.legend(custom_lines, ['X-axis', 'Y-axis', 'Z-axis', 'Trace'], loc='upper left')
    plt.show()


def plot_transformation_frames_plotly(frames: torch.Tensor, 
                                      center_first: bool = True, 
                                      scale: float = 1.0,
                                      show_trace: bool = True,
                                      title: str = "Interactive Transformation Frames"):
    """
    Plots a set of transformation frames in 3D using Plotly for interactivity.
    """
    if go is None:
        print("Error: Plotly is not installed.")
        return

    frames_np = _prepare_frames(frames, center_first)
    fig = go.Figure()

    origins = frames_np[:, :3, 3]

    # Add Trace
    if show_trace:
        fig.add_trace(go.Scatter3d(
            x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
            mode='lines+markers',
            line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=3, color='black'),
            name='Trace'
        ))

    for i, frame in enumerate(frames_np):
        origin = frame[:3, 3]
        for j, color in enumerate(['red', 'green', 'blue']):
            axis = frame[:3, j] * scale
            fig.add_trace(go.Scatter3d(
                x=[origin[0], origin[0] + axis[0]],
                y=[origin[1], origin[1] + axis[1]],
                z=[origin[2], origin[2] + axis[2]],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Frame {i} - {"XYZ"[j]}',
                showlegend=False
            ))
        
        # Add labels
        fig.add_trace(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]],
            mode='text',
            text=[str(i)],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
