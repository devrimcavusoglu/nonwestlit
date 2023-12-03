from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any, List

import numpy as np
from transformers import EvalPrediction


def sigmoid(ar: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-ar))


class ClassificationEvaluator(ABC):
    def __init__(self, num_labels: int):
        self.num_labels = num_labels

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        pass


class SingleLabelClassificationEvaluator(ClassificationEvaluator):
    def compute(self, input_data: EvalPrediction) -> Dict[str, Any]:
        pred_cls = input_data.predictions.argmax(-1)
        metrics = {}
        metrics["val_accuracy"] = sum(pred_cls == input_data.label_ids) / len(pred_cls)
        for i in range(self.num_labels):
            hits = sum(np.where(pred_cls == i, 1, 0) == np.where(input_data.label_ids == i, 1, -1))
            if sum(pred_cls == i) != 0:
                metrics[f"val_precision_{i}"] = hits / sum(pred_cls == i)
            else:
                metrics[f"val_precision_{i}"] = 0
            if sum(input_data.label_ids == i) != 0:
                metrics[f"val_recall_{i}"] = hits / sum(input_data.label_ids == i)
            else:
                metrics[f"val_recall_{i}"] = 0
            if metrics[f"val_precision_{i}"] + metrics[f"val_recall_{i}"] != 0:
                metrics[f"val_f1_{i}"] = (
                    2
                    * metrics[f"val_precision_{i}"]
                    * metrics[f"val_recall_{i}"]
                    / (metrics[f"val_precision_{i}"] + metrics[f"val_recall_{i}"])
                )
            else:
                metrics[f"val_f1_{i}"] = 0
        metrics["val_f1_macro"] = (metrics["val_f1_0"] + metrics["val_f1_1"] + metrics["val_f1_2"]) / 3
        gt_counts = Counter(input_data.label_ids.tolist())
        metrics["val_f1_weighted"] = (
            (
                gt_counts[0] * metrics["val_f1_0"]
                + gt_counts[1] * metrics["val_f1_1"]
                + gt_counts[2] * metrics["val_f1_2"]
            )
            / 3
            / len(input_data.label_ids)
        )
        return metrics


class MultiLabelClassificationEvaluator(ClassificationEvaluator):
    def __init__(self, threshold: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    @staticmethod
    def _set_thresholds(thresholds: float | List[float]) -> List[float]:
        if thresholds is None:
            thresholds = np.arange(0, 1, 0.05)
        elif isinstance(thresholds, float):
            thresholds = [thresholds]
        return thresholds

    def mean_average_precision(self, logits: np.ndarray, labels: np.ndarray, thresholds: float | List[float] = None):
        thresholds = self._set_thresholds(thresholds)
        slogits = sigmoid(logits)
        precisions_at_t = []
        for t in thresholds:
            pred_cls = np.where(slogits > t, 1, 0)
            pred_support = np.sum(pred_cls, axis=0)
            hits = pred_cls == labels
            precision = np.sum(np.logical_and(pred_cls == 1, hits), axis=0) / pred_support
            precision = np.nan_to_num(precision)  # convert nan to zero
            precisions_at_t.append(np.mean(precision))
        return np.mean(precisions_at_t)

    def mean_average_recall(self, logits: np.ndarray, labels: np.ndarray, thresholds: float | List[float] = None):
        thresholds = self._set_thresholds(thresholds)
        slogits = sigmoid(logits)
        recalls_at_t = []
        gt_support = np.sum(labels, axis=0)
        for t in thresholds:
            pred_cls = np.where(slogits > t, 1, 0)
            hits = pred_cls == labels
            recall = np.sum(np.logical_and(labels == 1, hits), axis=0) / gt_support
            recall = np.nan_to_num(recall)  # convert nan to zero
            recalls_at_t.append(np.mean(recall))
        return np.mean(recalls_at_t)

    def mean_average_accuracy(self, logits: np.ndarray, labels: np.ndarray, thresholds: float | List[float] = None):
        thresholds = self._set_thresholds(thresholds)
        slogits = sigmoid(logits)
        accs_at_t = []
        for t in thresholds:
            pred_cls = np.where(slogits > t, 1, 0)
            hits = pred_cls == labels
            accs_at_t.append(np.mean(hits))
        return np.mean(accs_at_t)

    def compute(self, input_data: EvalPrediction) -> Dict[str, Any]:
        logits = input_data.predictions
        labels = input_data.label_ids
        metrics = {}

        metrics["AA@0.5"] = self.mean_average_accuracy(logits, labels, thresholds=0.5)
        metrics["mAA"] = self.mean_average_accuracy(logits, labels)

        # AP (Average Precision)
        metrics["AP@0.5"] = self.mean_average_precision(logits, labels, thresholds=0.5)
        metrics["mAP"] = self.mean_average_precision(logits, labels)

        # AR (Average Recall)
        metrics["AR@0.5"] = self.mean_average_recall(logits, labels, thresholds=0.5)
        metrics["mAR"] = self.mean_average_recall(logits, labels)

        return metrics
