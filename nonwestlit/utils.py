import json
from typing import Optional

Nullable = Optional  # Semantically separated nullable type hint for return types.


def read_json(path: str):
    with open(path, "r") as fd_in:
        return json.load(fd_in)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Adopted from
    https://python.plainenglish.io/instruct-fine-tuning-falcon-7b-using-lora-6f79c3f234b0
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
