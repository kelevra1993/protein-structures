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


def print_shape(tensor: torch.Tensor, name: Optional[str] = ""):
    print(f"Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_list(tensor, round=4):
    print(np.round(tensor.tolist(), round))
