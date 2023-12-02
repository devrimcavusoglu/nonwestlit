from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Tuple

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from nonwestlit.utils import Nullable


class NonwestlitArticleTypes(StrEnum):
    cat_0 = "literary_text"
    cat_1 = "cultural_discourse"
    cat_2 = "other"


@dataclass
class NonwestlitBaseDataCollator(ABC):
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
    max_sequence_length: int = None
    is_mapping: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        inputs, labels = self.preprocess_inputs(features)
        model_inputs = self._tokenize(inputs)
        model_inputs = self.postprocess_inputs(model_inputs, labels)
        return model_inputs

    def _tokenize(self, inputs: List[str]) -> BatchEncoding:
        return_tensors = None if self.is_mapping else "pt"
        return self.tokenizer(
            inputs,
            truncation=True,
            return_tensors=return_tensors,
            padding=True,
            max_length=self.max_sequence_length,
            return_overflowing_tokens=False,
        )

    def preprocess_inputs(
        self, features: List[Dict[str, Any]]
    ) -> Tuple[List[str], Nullable[List[int]]]:
        """Preprocess given inputs"""
        if not self.is_mapping:
            inputs = [instance["input_ids"] for instance in features]
            if "labels" in features[0]:
                labels = [instance["labels"] for instance in features]
            else:
                labels = None
        else:
            inputs = features["input_ids"]
            labels = features.get("labels", None)
        return inputs, labels

    def postprocess_inputs(self, model_inputs: BatchEncoding, labels: List[int]) -> BatchEncoding:
        """Postprocess tokenized model inputs"""
        if "overflow_to_sample_mapping" in model_inputs:
            model_inputs.pop("overflow_to_sample_mapping")
        return model_inputs


@dataclass
class NonwestlitCausalLMDataCollator(NonwestlitBaseDataCollator):
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

    def postprocess_inputs(self, model_inputs: BatchEncoding, labels: List[int]) -> BatchEncoding:
        model_inputs = super().postprocess_inputs(model_inputs, labels)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


@dataclass
class NonwestlitPromptTuningDataCollator(NonwestlitBaseDataCollator):
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

    num_virtual_tokens: int = None

    def _tokenize(self, inputs: List[str]) -> BatchEncoding:
        if self.max_sequence_length is not None:
            self.max_sequence_length -= self.num_virtual_tokens

    def preprocess_inputs(
        self, features: List[Dict[str, Any]]
    ) -> Tuple[List[str], Nullable[List[int]]]:
        inputs = [
            f"""INPUT_TEXT: {instance['input_ids']} LABEL: {NonwestlitArticleTypes[f'cat_{instance["labels"]}']}"""
            for instance in features
        ]
        return inputs, None

    def postprocess_inputs(self, model_inputs: BatchEncoding, labels: List[int]) -> BatchEncoding:
        model_inputs = super().postprocess_inputs(model_inputs, labels)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs


@dataclass
class NonwestlitSequenceClassificationDataCollator(NonwestlitBaseDataCollator):
    """
    Collator for Prompt Tuning/Learning task. The collator prepares the batch input text
        as follows:
        - input_ids: <input-article-text>
        - attention_mask: <attention_mask>
        - labels: <label_id>

        Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer object for the handling of the inputs.
    """

    def postprocess_inputs(self, model_inputs: BatchEncoding, labels: List[int]) -> BatchEncoding:
        if "overflow_to_sample_mapping" in model_inputs:
            overflow_to_sample_mapping: List = model_inputs.pop("overflow_to_sample_mapping").tolist()
            all_labels = []
            for i in range(len(labels)):
                sample_labels = [labels[i]] * overflow_to_sample_mapping.count(i)
                all_labels.extend(sample_labels)
            model_inputs["labels"] = torch.tensor(all_labels)
        elif isinstance(labels, int):
            model_inputs["labels"] = torch.tensor(labels)
        else:
            model_inputs["labels"] = torch.tensor(labels, dtype=torch.float16)
        return model_inputs


if __name__ == "__main__":
    import datasets
    from transformers import AutoTokenizer

    from nonwestlit import PROJECT_ROOT

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    max_sequence_length = 1024
    num_labels = 8
    data_path = PROJECT_ROOT / "data/ottoman_second_level"
    d = datasets.load_dataset(data_path.as_posix(), "cultural_discourse_subject", split="train",
                              tokenizer=tokenizer)
    collator = NonwestlitSequenceClassificationDataCollator(
        tokenizer, max_sequence_length, is_mapping=True,
    )
    d = d.map(collator)
    for item in d:
        print(item)
        break
