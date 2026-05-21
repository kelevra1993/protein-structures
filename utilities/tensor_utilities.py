import torch
import numpy as np
from typing import Optional

from torch import nn


def get_device() -> torch.device:
    """
    todo to be later documented
    :return:
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_tensor_shape(tensor: torch.Tensor, name: Optional[str] = ""):
    """Todo add documentation"""
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_list(tensor, round=4):
    """Todo add documentation"""
    print(np.round(tensor.tolist(), round))


def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = ""):
    """Todo add documentation"""
    print(f"Tensor {name} Is Of Type : {tensor.dtype}")


def unsqueeze_tensor(input: torch.Tensor, direction: str, number: int = 1) -> torch.Tensor:
    """
    todo add documentation
    """
    if direction not in ["left", "right"]:
        print("Warning : direction must either be left or right")
        raise NotImplementedError

    for i in range(number):
        input = input.unsqueeze(dim=-1 if direction == "right" else 0)

    return input


def specialised_one_hot_encoder(input_tensor: torch.Tensor,
                                bin_tensor: torch.Tensor) -> torch.Tensor:
    """Todo Add documentaiton"""

    # Add A Dummy Dimension At The End Of Input Tensor For Broadcasting
    expanded_input = input_tensor.unsqueeze(-1)

    # Compute Absolute Difference : Broadcasts (r, r, 1) and (bins,) into (r, r, bins)
    # This computes the distance from EVERY input to EVERY bin simultaneously.
    differences = torch.abs(expanded_input - bin_tensor)

    # Find The Index Of The Minimum Difference : Closest Bin
    indices = torch.argmin(differences, dim=-1)

    # Generate One Hot Encoding
    output = nn.functional.one_hot(indices, num_classes=bin_tensor.size(-1))

    # Todo properly specify the output dimension
    return output
