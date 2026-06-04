import json


def read_json(path: str):
    with open(path, "r") as file:
        json_data = json.loads(file.read())

    return json_data
