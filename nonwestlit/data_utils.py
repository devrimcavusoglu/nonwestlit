from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import numpy as np
from transformers import PreTrainedTokenizerBase

from nonwestlit.collator import (
    NonwestlitBaseDataCollator,
    NonwestlitCausalLMDataCollator,
    NonwestlitPromptTuningDataCollator,
    NonwestlitSequenceClassificationDataCollator,
)
from nonwestlit.dataset import NONWESTLITDataset
from nonwestlit.utils import NonwestlitTaskTypes


def get_collator(
    task_type: str,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
    is_mapping: bool = True,
    num_virtual_tokens: int | None = None,
):
    if task_type in [NonwestlitTaskTypes.seq_cls, NonwestlitTaskTypes.multi_seq_cls]:
        return NonwestlitSequenceClassificationDataCollator(
            tokenizer, max_sequence_length, is_mapping=is_mapping
        )
    elif task_type == NonwestlitTaskTypes.casual_lm:
        return NonwestlitCausalLMDataCollator(tokenizer, max_sequence_length, is_mapping=is_mapping)
    elif task_type == NonwestlitTaskTypes.prompt_tuning:
        return NonwestlitPromptTuningDataCollator(
            tokenizer, num_virtual_tokens, max_sequence_length, is_mapping=is_mapping
        )


def load_torch_data(
    train_data_path: str | None = None,
    eval_data_path: str | None = None,
    test_data_path: str | None = None,
) -> Tuple[NONWESTLITDataset | None, NONWESTLITDataset | None, NONWESTLITDataset | None]:
    train_dataset = NONWESTLITDataset(train_data_path) if train_data_path is not None else None
    eval_dataset = NONWESTLITDataset(eval_data_path) if eval_data_path is not None else None
    test_dataset = NONWESTLITDataset(test_data_path) if test_data_path is not None else None
    return train_dataset, eval_dataset, test_dataset


def load_hf_data(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    collator: NonwestlitBaseDataCollator | None = None,
    splits: list[str] = ["train", "validation"],
    max_sequence_length: int | None = None,
    **kwargs
):
    data_path = Path(data_path)
    if ":" in data_path.name:
        name, subset = data_path.name.split(":")
        name = data_path.parent / name.strip()
        subset = subset.strip()
    else:
        name, subset = data_path, None
    d = datasets.load_dataset(
        name.as_posix(), subset, tokenizer=tokenizer, max_sequence_length=max_sequence_length, **kwargs
    )
    train_dataset, eval_dataset, test_dataset = None, None, None

    if "train" in splits:
        train_dataset = d["train"]
        train_dataset = train_dataset.map(collator)

    if "validation" in splits:
        eval_dataset = d["validation"]
        eval_dataset = eval_dataset.map(collator)

    if "test" in splits:
        test_dataset = d["test"]
        test_dataset = test_dataset.map(collator)

    return train_dataset, eval_dataset, test_dataset


def _to_one_hot(labels: list[int], num_labels: int) -> list[int]:
    x = np.zeros(num_labels, dtype=np.float16)
    x[labels] = 1.0
    return x.tolist()


def _get_label(article: Dict[str, Any], multi_label: bool, num_labels: int):
    label = article.get("label")
    if label is None:
        if not multi_label:
            return
        else:
            label = _to_one_hot(labels=[], num_labels=num_labels)
    elif isinstance(label, int):
        label = int(label) - 1
    elif isinstance(label, str):
        if multi_label:
            label = [int(l) - 1 for l in label.split(",")]
            label = _to_one_hot(label, num_labels)
        else:
            label = int(label) - 1
    return label
