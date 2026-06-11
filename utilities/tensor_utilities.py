import torch
import numpy as np
from typing import Optional

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
    todo add documentation
    """
    print_blue(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    todo add documentation
    """
    print_yellow(f"Tensor {name} Is Of Type : {tensor.dtype}")


def print_tensor_device(tensor: torch.Tensor, name: Optional[str] = ""):
    """
    todo add documentation
    """
    print_green(f"Tensor {name} Is On : {tensor.device}")


def print_tensor_status(tensor: torch.Tensor, name: Optional[str] = ""):
    """ todo add documenetation"""
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
