import json
import os
from configparser import ConfigParser
from enum import StrEnum
from typing import Optional

import neptune
import numpy as np
from neptune import Run
from transformers.integrations import NeptuneCallback

from nonwestlit import NEPTUNE_CFG

Nullable = Optional  # Semantically separated nullable type hint for return types.


class NonwestlitTaskTypes(StrEnum):
    multi_seq_cls = "multilabel-sequence-classification"
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


# TENSOR UTILS #


def sigmoid(ar: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-ar))


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def geometric_mean(x: np.ndarray) -> np.ndarray:
    return np.prod(x) ** (1 / len(x))


# EXPERIMENT TRACKING UTILS #


def _check_neptune_creds(neptune_project_name: str):
    if os.getenv("NEPTUNE_PROJECT") is not None and os.getenv("NEPTUNE_API_TOKEN") is not None:
        return
    elif NEPTUNE_CFG.exists():
        cfg = read_cfg(NEPTUNE_CFG.as_posix())
        neptune_project = cfg[neptune_project_name]["project"]
        neptune_token = cfg[neptune_project_name]["api_token"]
        assert neptune_project is not None and neptune_token is not None
        # set as environment variables
        os.environ["NEPTUNE_PROJECT"] = neptune_project
        os.environ["NEPTUNE_API_TOKEN"] = neptune_token
        return
    raise EnvironmentError("Neither environment variables nor neptune.cfg is found.")


def create_neptune_run(
    neptune_project_name: str, experiment_tracking: bool, callbacks: list | None = None
) -> Nullable[Run]:
    if not experiment_tracking:
        return None
    _check_neptune_creds(neptune_project_name)
    run = neptune.init_run()
    if callbacks is not None:
        neptune_callback = NeptuneCallback(run=run)
        callbacks.append(neptune_callback)
    return run
