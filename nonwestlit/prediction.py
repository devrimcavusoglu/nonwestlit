from typing import List, Tuple, Dict

from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, TextClassificationPipeline, PreTrainedTokenizerBase, BatchEncoding
from transformers.pipelines.base import GenericTensor

from nonwestlit.dataset import NONWESTLITDataset


class NONWESTLITClassificationPipeline(TextClassificationPipeline):
    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, top_k="", **tokenizer_kwargs):
        tokenizer_kwargs["return_overflowing_tokens"] = tokenizer_kwargs.get("return_overflowing_tokens", True)
        return super()._sanitize_parameters(return_all_scores, function_to_apply, top_k, **tokenizer_kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        pass