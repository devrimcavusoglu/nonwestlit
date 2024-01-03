import numpy as np
from transformers import BatchEncoding, TextClassificationPipeline
from transformers.pipelines.text_classification import sigmoid, softmax
from transformers.utils import ModelOutput

from nonwestlit.utils import NonwestlitTaskTypes


class NONWESTLITClassificationPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overflow_to_sample_mapping = None

    def _sanitize_parameters(
        self,
        return_all_scores=None,
        function_to_apply=None,
        top_k="",
        task_type: str = "sequence-classification",
        max_sequence_length: int = 2048,
        return_scores_only: bool = None,
        in_sample_batch_size: int = None,
        **tokenizer_kwargs
    ):
        tokenizer_kwargs["return_overflowing_tokens"] = tokenizer_kwargs.get(
            "return_overflowing_tokens", True
        )
        tokenizer_kwargs["max_length"] = tokenizer_kwargs.get("max_length", max_sequence_length)
        tokenizer_kwargs["truncation"] = tokenizer_kwargs.get("truncation", True)
        tokenizer_kwargs["padding"] = tokenizer_kwargs.get("padding", True)
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = super()._sanitize_parameters(
            return_all_scores, function_to_apply, top_k, **tokenizer_kwargs
        )
        if task_type == NonwestlitTaskTypes.seq_cls:
            self.model.config.problem_type = "single_label_classification"
        elif task_type == NonwestlitTaskTypes.multi_seq_cls:
            self.model.config.problem_type = "multi_label_classification"

        postprocess_kwargs["return_scores_only"] = return_scores_only
        forward_kwargs["in_sample_batch_size"] = in_sample_batch_size

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def batch_iterator(self, inputs, batch_size: int):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        start = 0
        end = min(batch_size, len(input_ids))
        step = batch_size

        while end <= len(input_ids):
            yield BatchEncoding(
                data={
                    "input_ids": input_ids[start:end, ...],
                    "attention_mask": attention_mask[start:end, ...],
                }
            )
            if end == len(input_ids):
                break
            start += step
            if end + step > len(input_ids):
                end = len(input_ids)
            else:
                end += step

    def merge_batch_outputs(self, all_outputs: list[ModelOutput]) -> ModelOutput:
        logits = []
        for output in all_outputs:
            logits.append(output["logits"].cpu().detach().numpy())
        return ModelOutput(
            logits=np.vstack(logits),
        )

    def _forward(self, model_inputs, in_sample_batch_size: int = 1):
        self.overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        logits = []
        for batch in self.batch_iterator(model_inputs, in_sample_batch_size):
            model_output = super()._forward(model_inputs=batch)
            logits.append(model_output["logits"].cpu().detach().numpy())
        return ModelOutput(
            logits=np.vstack(logits),
        )

    def postprocess(
        self, model_outputs, function_to_apply=None, top_k=1, _legacy=True, return_scores_only=False
    ):
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification":
                function_to_apply = sigmoid
            elif self.model.config.problem_type == "single_label_classification":
                function_to_apply = softmax
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply

        logits = model_outputs["logits"]
        if function_to_apply is not None:
            probs = function_to_apply(logits)
        else:
            probs = logits

        # pooling
        scores = probs.mean(0)
        if return_scores_only:
            return scores

        if top_k == 1 and _legacy:
            return {
                "label": self.model.config.id2label[scores.argmax().item()],
                "score": scores.max().item(),
            }

        dict_scores = [
            {"label": self.model.config.id2label[i], "score": score.item()}
            for i, score in enumerate(scores)
        ]
        if not _legacy:
            dict_scores.sort(key=lambda x: x["score"], reverse=True)
            if top_k is not None:
                dict_scores = dict_scores[:top_k]
        return dict_scores
