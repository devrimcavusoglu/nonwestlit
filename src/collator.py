from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class NONWESTLITDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for instance in features:
            tokenized_input = self.tokenizer(
                    instance["input_ids"], truncation=True, return_tensors=self.return_tensors
            )
            batch["input_ids"].append(tokenized_input["input_ids"])
            batch["attention_mask"].append(tokenized_input["attention_mask"])
            batch["labels"].append(instance["labels"])

        for k, v in batch.items():
            if k == "labels":
                batch[k] = torch.tensor(v).view(-1, 1)
                continue
            batch[k] = torch.cat(v)
        return batch
