import json
from typing import Optional

Nullable = Optional  # Semantically separated nullable type hint for return types.


def read_json(path: str):
    with open(path, "r") as fd_in:
        return json.load(fd_in)
