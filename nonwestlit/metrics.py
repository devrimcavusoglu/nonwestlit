from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from transformers import EvalPrediction

from nonwestlit.utils import sigmoid, softmax


class ClassificationMetrics(ABC):
    def __init__(self, num_labels: int):
        self.num_labels = num_labels

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        pass


class SingleLabelClassificationMetrics(ClassificationMetrics):
    def compute(self, input_data: EvalPrediction, prefix: str = "val", function_to_apply: Callable = softmax) -> Dict[str, Any]:
        probs = input_data.predictions
        if function_to_apply is not None:
            probs = function_to_apply(probs)
        pred_cls = probs.argmax(-1)
        metrics = {}
        metrics[f"{prefix}_accuracy"] = sum(pred_cls == input_data.label_ids) / len(pred_cls)
        for i in range(self.num_labels):
            hits = sum(np.where(pred_cls == i, 1, 0) == np.where(input_data.label_ids == i, 1, -1))
            if sum(pred_cls == i) != 0:
                metrics[f"{prefix}_precision_{i}"] = hits / sum(pred_cls == i)
            else:
                metrics[f"{prefix}_precision_{i}"] = 0
            if sum(input_data.label_ids == i) != 0:
                metrics[f"{prefix}_recall_{i}"] = hits / sum(input_data.label_ids == i)
            else:
                metrics[f"{prefix}_recall_{i}"] = 0
            if metrics[f"{prefix}_precision_{i}"] + metrics[f"val_recall_{i}"] != 0:
                metrics[f"{prefix}_f1_{i}"] = (
                    2
                    * metrics[f"{prefix}_precision_{i}"]
                    * metrics[f"{prefix}_recall_{i}"]
                    / (metrics[f"{prefix}_precision_{i}"] + metrics[f"{prefix}_recall_{i}"])
                )
            else:
                metrics[f"{prefix}_f1_{i}"] = 0
        metrics[f"{prefix}_f1_macro"] = (
            metrics[f"{prefix}_f1_0"] + metrics[f"{prefix}_f1_1"] + metrics[f"{prefix}_f1_2"]
        ) / 3
        gt_counts = Counter(input_data.label_ids.tolist())
        metrics[f"{prefix}_f1_weighted"] = (
            (
                gt_counts[0] * metrics[f"{prefix}_f1_0"]
                + gt_counts[1] * metrics[f"{prefix}_f1_1"]
                + gt_counts[2] * metrics[f"{prefix}_f1_2"]
            )
            / 3
            / len(input_data.label_ids)
        )
        return metrics


class MultiLabelClassificationMetrics(ClassificationMetrics):
    @staticmethod
    def _set_thresholds(thresholds: float | List[float]) -> List[float]:
        if thresholds is None:
            thresholds = np.arange(0, 1, 0.05)
        elif isinstance(thresholds, float):
            thresholds = [thresholds]
        return thresholds

    def prf_at_threshold(
        self, probs: np.ndarray, labels: np.ndarray, thresholds: float | List[float] = None
    ):
        thresholds = self._set_thresholds(thresholds)
        metrics_at_t = []
        for t in thresholds:
            pred_cls = np.where(probs >= t, 1, 0)
            p, r, f, _ = precision_recall_fscore_support(y_pred=pred_cls, y_true=labels, average="micro")
            metrics_at_t.append([p, r, f])
        return np.mean(metrics_at_t, axis=0)

    def compute(
        self, input_data: EvalPrediction, function_to_apply: Callable = sigmoid
    ) -> Dict[str, Any]:
        probs = input_data.predictions
        if function_to_apply is not None:
            probs = function_to_apply(input_data.predictions)
        labels = input_data.label_ids
        metrics = {}

        p_t, r_t, f_t = self.prf_at_threshold(probs, labels, thresholds=0.5)
        pmp, pmr, pmf = self.prf_at_threshold(probs, labels)

        metrics["AP@0.5"] = p_t
        metrics["AR@0.5"] = r_t
        metrics["AF1@0.5"] = f_t

        metrics["PMP"] = pmp
        metrics["PMR"] = pmr
        metrics["PMF"] = pmf

        metrics["mAP"] = average_precision_score(y_true=labels, y_score=probs)

        return metrics
