from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, TextClassificationPipeline
from transformers.pipelines.text_classification import sigmoid, softmax

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
        return_scores_only: bool = False,
        **tokenizer_kwargs
    ):
        tokenizer_kwargs["return_overflowing_tokens"] = tokenizer_kwargs.get(
            "return_overflowing_tokens", True
        )
        tokenizer_kwargs["max_length"] = tokenizer_kwargs.get("max_length", max_sequence_length)
        tokenizer_kwargs["truncation"] = tokenizer_kwargs.get("truncation", True)
        tokenizer_kwargs["padding"] = tokenizer_kwargs.get("padding", True)
        if task_type == NonwestlitTaskTypes.seq_cls:
            self.model.config.problem_type = "single_label_classification"
        elif task_type == NonwestlitTaskTypes.multi_seq_cls:
            self.model.config.problem_type = "multi_label_classification"

        self.return_scores_only = return_scores_only

        return super()._sanitize_parameters(
            return_all_scores, function_to_apply, top_k, **tokenizer_kwargs
        )

    def _forward(self, model_inputs):
        self.overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        return super()._forward(model_inputs=model_inputs)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification":
                function_to_apply = sigmoid
            elif self.model.config.problem_type == "single_label_classification":
                function_to_apply = softmax
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply

        logits = model_outputs["logits"].numpy()
        if function_to_apply is not None:
            probs = function_to_apply(logits)
        else:
            probs = logits

        # pooling
        scores = probs.mean(0)
        if self.return_scores_only:
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
