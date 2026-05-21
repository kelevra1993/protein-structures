import torch
import numpy as np
from typing import Optional


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
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))

def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = ""):
    print(f"Tensor {name} Is Of Type : {tensor.dtype}")

def unsqueeze_tensor(input: torch.Tensor, direction: str, number: int=1) -> torch.Tensor:
    """
    todo add documentation
    """
    if direction not in ["left", "right"]:
        print("Warning : direction must either be left or right")
        raise NotImplementedError

    for i in range(number):
        input = input.unsqueeze(dim=-1 if direction == "right" else 0)

    return input