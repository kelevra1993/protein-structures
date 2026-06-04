import json
import numpy as np


def read_npz_file(path: str):
    return np.load(path, allow_pickle=True)


def read_json(path: str):
    with open(path, "r") as file:
        json_data = json.loads(file.read())

    return json_data
