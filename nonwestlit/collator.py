from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


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
        batch["labels"] = batch["input_ids"].copy()
        return batch


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
