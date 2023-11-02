from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding


class NonwestlitArticleTypes(StrEnum):
    cat_0 = "literary_text"
    cat_1 = "cultural_discourse"
    cat_2 = "other"


@dataclass
class NonwestlitCausalLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"input_ids": [], "attention_mask": []}
        for instance in features:
            tokenized_input = self.tokenizer(
                instance["input_ids"], truncation=True, return_tensors=self.return_tensors, padding=True
            )
            batch["input_ids"].append(tokenized_input["input_ids"])
            batch["attention_mask"].append(tokenized_input["attention_mask"])

        for k, v in batch.items():
            batch[k] = torch.cat(v)
        # Set labels which is the same as the input, the shifting indices operation
        # is done in the modeling part, so we just copy the inputs.
        batch["labels"] = batch["input_ids"].clone()
        return batch


@dataclass
class NonwestlitPromptTuningDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        inputs = [f"INPUT_TEXT: {instance['input_ids']} LABEL: " for instance in features]
        model_inputs = self.tokenizer(
                inputs, truncation=True, return_tensors=self.return_tensors, padding=True
        )
        text_labels = [NonwestlitArticleTypes[f"cat_{instance['labels']}"] for instance in features]
        labels = self.tokenizer(text_labels, padding=True, return_tensors=self.return_tensors)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


@dataclass
class NonwestlitSequenceClassificationDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for instance in features:
            tokenized_input = self.tokenizer(
                instance["input_ids"], truncation=True, return_tensors=self.return_tensors, padding=True
            )
            batch["input_ids"].append(tokenized_input["input_ids"])
            batch["attention_mask"].append(tokenized_input["attention_mask"])
            batch["labels"].append(instance["labels"])

        for k, v in batch.items():
            if k == "labels":
                batch[k] = torch.tensor(v)
                continue
            batch[k] = torch.cat(v)
        return batch
