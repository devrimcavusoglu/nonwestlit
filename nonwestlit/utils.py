import json
from configparser import ConfigParser
from enum import StrEnum
from typing import Optional

Nullable = Optional  # Semantically separated nullable type hint for return types.


class NonwestlitTaskTypes(StrEnum):
    seq_cls = "sequence-classification"
    casual_lm = "causal-lm"
    prompt_tuning = "prompt-tuning"


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fd_in:
        return json.load(fd_in)


def read_cfg(path: str) -> ConfigParser:
    cfg = ConfigParser()
    cfg.read(path)
    return cfg


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
