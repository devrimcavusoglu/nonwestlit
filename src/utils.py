import json


def read_json(path: str):
    with open(path, "r") as fd_in:
        return json.load(fd_in)
