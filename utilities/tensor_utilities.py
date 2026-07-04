import torch
import numpy as np
from typing import Optional, Any

from torch import nn

from utilities.os_utilities import print_blue, print_yellow, print_green


def get_device() -> torch.device:
    """
    Identifies and returns the most efficient available hardware device for tensor computations.

    This utility ensures that the project remains cross-platform compatible by prioritizing
    CUDA (NVIDIA GPUs), then MPS (Apple Silicon GPUs), and falling back to CPU if no
    accelerators are detected. It is used globally across all modules to maintain
    device consistency.

    :return: The detected torch.device (e.g., 'cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_tensor_shape(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    Logs the shape of a tensor to the console in a formatted blue string.

    In the global project context, this utility is used during development and debugging
    to verify that tensor dimensions align with expected shapes (e.g., MSA or Pair representations)
    after complex transformations or contractions.

    :param tensor: The torch.Tensor whose shape will be printed. Shape: (*).
    :param name: An optional label to identify the tensor in the output.
    """
    print_blue(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    Logs the data type of a tensor to the console in a formatted yellow string.

    Ensures that tensors maintain consistent dtypes (e.g., torch.float32) across different
    architectural modules, preventing type mismatch errors during multi-platform
    execution (CPU/CUDA/MPS).

    :param tensor: The torch.Tensor whose dtype will be printed. Shape: (*).
    :param name: An optional label to identify the tensor in the output.
    """
    print_yellow(f"Tensor {name} Is Of Type : {tensor.dtype}")


def print_tensor_device(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    Logs the hardware device of a tensor to the console in a formatted green string.

    Crucial for identifying and resolving device placement issues, ensuring all tensors
    participating in an operation reside on the same hardware (CUDA, MPS, or CPU) as
    mandated by the project's cross-platform compatibility guidelines.

    :param tensor: The torch.Tensor whose device will be printed. Shape: (*).
    :param name: An optional label to identify the tensor in the output.
    """
    print_green(f"Tensor {name} Is On : {tensor.device}")


def print_tensor_status(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    Provides a comprehensive log of a tensor's shape, type, and device.

    Aggregates individual printing utilities to offer a single-point snapshot of a tensor's
    state. This is particularly useful for deep debugging within dense modules like the
    Evoformer or Structure Module where multiple transformations occur.

    :param tensor: The torch.Tensor to inspect. Shape: (*).
    :param name: An optional label to identify the tensor in the output.
    """
    print_tensor_shape(tensor=tensor, name=name)
    print_tensor_type(tensor=tensor, name=name)
    print_tensor_device(tensor=tensor, name=name)


def print_tensor_list(tensor: torch.Tensor, round: int = 4):
    """
    Converts a tensor to a list and prints it with specified rounding precision.

    Useful for inspecting the numerical values of small tensors or intermediate results
    during the development and testing of architectural components.

    :param tensor: The torch.Tensor to print.
    :param round: Number of decimal places for rounding.
    """
    print(np.round(tensor.tolist(), round))


def unsqueeze_tensor(input: torch.Tensor, direction: str, number: int = 1) -> torch.Tensor:
    """
    Expands the dimensions of a tensor by adding singleton dimensions at the specified end.

    This utility is frequently used to prepare tensors for broadcasting or to match
    the expected input shapes of various neural network layers in the AlphaFold II
    architecture.

    :param input: The source torch.Tensor.
    :param direction: Where to add dimensions; must be "left" (start) or "right" (end).
    :param number: The number of dimensions to add.
    :return: The transformed torch.Tensor with 'number' additional dimensions.
             - If direction is "left": shape (1, ..., input.shape)
             - If direction is "right": shape (input.shape, ..., 1)
    """
    if direction not in ["left", "right"]:
        print("Warning : direction must either be left or right")
        raise NotImplementedError

    for i in range(number):
        input = input.unsqueeze(dim=-1 if direction == "right" else 0)

    return input


def specialised_one_hot_encoder(input_tensor: torch.Tensor,
                                bin_tensor: torch.Tensor) -> torch.Tensor:
    """
    Encodes continuous or discrete values into a one-hot representation based on a set of bins.

    This function finds the closest bin for each value in the input tensor and returns
     a one-hot vector representing that bin. In the global project context, it is critical
    for 'Relative Position Encoding', where the distance between residues is discretized
    into fixed bins before being embedded into the pair representation.

    :param input_tensor: A tensor of arbitrary shape (e.g., [number_residues, number_residues]) 
                         containing values to be binned.
    :param bin_tensor: A 1D tensor of shape [number_bins] defining the center or values of the bins.
    :return: A one-hot encoded tensor of shape [*input_tensor.shape, number_bins].
             For residue distance encoding, the shape is [number_residues, number_residues, number_bins].
    """

    # Add A Dummy Dimension At The End Of Input Tensor For Broadcasting
    expanded_input = input_tensor.unsqueeze(-1)

    # Compute Absolute Difference : Broadcasts (r, r, 1) and (bins,) into (r, r, bins)
    # This computes the distance from EVERY input to EVERY bin simultaneously.
    differences = torch.abs(expanded_input - bin_tensor)

    # Find The Index Of The Minimum Difference : Closest Bin
    indices = torch.argmin(differences, dim=-1)

    # Generate One Hot Encoding
    output = nn.functional.one_hot(indices, num_classes=bin_tensor.size(-1))

    return output


def extract_angles(matrix: Any) -> np.ndarray:
    """
    Extracts degrees from an array of (cos, sin) pairs.

    This utility function fits into the global project context by converting the 
    neural network's continuous unnormalized or normalized (cos, sin) angle predictions 
    back into discrete degrees. This is commonly used in validation and logging 
    modules (such as `log_angles` in the Trainer) to provide human-readable angular metrics 
    for debugging the Structure Module.

    Args:
        matrix (Any): An array-like object (list, np.ndarray, torch.Tensor) containing pairs of (cos, sin) values.
            Expected shape: `(number_angles, 2)` where index 0 is cos and index 1 is sin.

    Returns:
        np.ndarray: A column vector of extracted angles in degrees.
            Shape: `(number_angles, 1)`.
    """
    # Ensure the input is treated as a NumPy array
    array_matrix = np.asarray(matrix)

    # Calculate angles in radians using arctan2(y, x)
    angles_radians = np.arctan2(array_matrix[:, 1], array_matrix[:, 0])

    # Convert the angles from radians to degrees
    angles_degrees = np.degrees(angles_radians)

    # Reshape and return as an n x 1 column vector
    return angles_degrees.reshape(-1, 1)