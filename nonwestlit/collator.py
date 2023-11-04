from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase


class NonwestlitArticleTypes(StrEnum):
    cat_0 = "literary_text"
    cat_1 = "cultural_discourse"
    cat_2 = "other"


@dataclass
class NonwestlitCausalLMDataCollator:
    """
    Collator for Prompt Tuning/Learning task. The collator prepares the batch input text
        as follows:
        - input_ids: <input-article-text>
        - attention_mask: <attention_mask>
        - labels: `input_ids`
    to be trained with Causal LM objective, so labels are the same with the input_ids, but shifted.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer object for the handling of the inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        inputs = [instance["input_ids"] for instance in features]
        model_inputs = self.tokenizer(
            inputs, truncation=True, return_tensors=self.return_tensors, padding=True
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


@dataclass
class NonwestlitPromptTuningDataCollator:
    """
    Collator for Prompt Tuning/Learning task. The collator prepares the batch input text
        as follows:
        - input_ids: "INPUT_TEXT: <input-article-text> LABEL: <label-name>"
        - attention_mask: <attention_mask>
        - labels: `input_ids`
    to be trained with Causal LM objective, so labels are the same with the input_ids, but shifted.

    Note:
        This collator truncates the input text and prompt based on `self.tokenizer.model_max_length`,
        so that with additional virtual tokens it doesn't excess the max input size.

        Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer object for the handling of the inputs.
        num_virtual_tokens (int): Number of virtual tokens for PromptTuning method.
    """

    tokenizer: PreTrainedTokenizerBase
    num_virtual_tokens: int
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        inputs = [
            f"""INPUT_TEXT: {instance['input_ids']} LABEL: {NonwestlitArticleTypes[f'cat_{instance["labels"]}']}"""
            for instance in features
        ]
        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            return_tensors=self.return_tensors,
            max_length=self.tokenizer.model_max_length - self.num_virtual_tokens,
            padding=True,
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


@dataclass
class NonwestlitSequenceClassificationDataCollator:
    """
    Collator for Prompt Tuning/Learning task. The collator prepares the batch input text
        as follows:
        - input_ids: <input-article-text>
        - attention_mask: <attention_mask>
        - labels: <label_id>

        Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer object for the handling of the inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        inputs = [instance["input_ids"] for instance in features]
        labels = [instance["labels"] for instance in features]
        model_inputs = self.tokenizer(
            inputs, truncation=True, return_tensors=self.return_tensors, padding=True
        )
        model_inputs["labels"] = torch.tensor(labels)
        return model_inputs
