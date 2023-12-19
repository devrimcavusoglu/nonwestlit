from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score, f1_score
from transformers import EvalPrediction

from nonwestlit.utils import sigmoid, softmax


class ClassificationMetrics(ABC):

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        pass


class SingleLabelClassificationMetrics(ClassificationMetrics):
    def one_hot_encoding(self, x: np.ndarray, n_labels: int) -> np.ndarray:
        m = len(x)
        ohe = np.zeros((m, n_labels))
        idx = np.arange(m)
        ohe[idx, x] = 1
        return ohe

    def compute(self, input_data: EvalPrediction, prefix: str = "val", function_to_apply: Callable = softmax) -> Dict[str, Any]:
        probs = input_data.predictions
        n_labels = probs.shape[1]
        if function_to_apply is not None:
            probs = function_to_apply(probs)
        pred_cls = probs.argmax(-1)
        metrics = {}
        metrics[f"{prefix}_accuracy"] = accuracy_score(y_true=input_data.label_ids, y_pred=pred_cls)
        metrics[f"{prefix}_f1_macro"] = f1_score(y_true=input_data.label_ids, y_pred=pred_cls, average="macro")
        metrics[f"{prefix}_f1_weighted"] = f1_score(y_true=input_data.label_ids, y_pred=pred_cls, average="weighted")
        p, r, f, _ = precision_recall_fscore_support(y_pred=pred_cls, y_true=input_data.label_ids, average=None)
        for i in range(len(p)):
            metrics[f"{prefix}_precision_{i}"] = p[i]
            metrics[f"{prefix}_recall_{i}"] = r[i]
            metrics[f"{prefix}_f1_{i}"] = f[i]

        labels_ohe = self.one_hot_encoding(input_data.label_ids, n_labels)
        metrics["mAP"] = average_precision_score(y_true=labels_ohe, y_score=probs)  # default avg='macro'
        metrics["mAP_weighted"] = average_precision_score(y_true=labels_ohe, y_score=probs, average="weighted")
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
        metrics["mAP_weighted"] = average_precision_score(y_true=labels, y_score=probs, average="weighted")

        return metrics


if __name__ == "__main__":
    np.random.seed(42)
    predictions = np.random.randn(9, 3)
    predictions = np.apply_along_axis(softmax, axis=1, arr=predictions)
    labels = np.random.randint(0, 3, size=9)
    print(predictions.argmax(-1))
    print(labels)
    metric_obj = EvalPrediction(label_ids=labels, predictions=predictions)
    metrics = SingleLabelClassificationMetrics(num_labels=3)
    out = metrics(metric_obj, function_to_apply=None)
    import json
    print(json.dumps(out, indent=2))
