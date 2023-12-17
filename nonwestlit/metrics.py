from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from transformers import EvalPrediction

from nonwestlit.utils import sigmoid


class ClassificationMetrics(ABC):
    def __init__(self, num_labels: int):
        self.num_labels = num_labels

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        pass


class SingleLabelClassificationMetrics(ClassificationMetrics):
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


class MultiLabelClassificationMetrics(ClassificationMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _set_thresholds(thresholds: float | List[float]) -> List[float]:
        if thresholds is None:
            thresholds = np.arange(0, 1, 0.05)
        elif isinstance(thresholds, float):
            thresholds = [thresholds]
        return thresholds

    def precision_recall_f1(self, pred_cls: np.ndarray, labels: np.ndarray):
        pred_support = np.sum(pred_cls, axis=0)
        gt_support = np.sum(labels, axis=0)
        hits = pred_cls == labels

        precision = np.sum(np.logical_and(pred_cls == 1, hits), axis=0) / pred_support
        precision = np.nan_to_num(precision)  # convert nan to zero

        recall = np.sum(np.logical_and(labels == 1, hits), axis=0) / gt_support
        recall = np.nan_to_num(recall)  # convert nan to zero

        pr = np.vstack([precision, recall])
        f1 = 2 * np.prod(pr, axis=0) / np.sum(pr, axis=0)
        f1 = np.nan_to_num(f1)

        ap = np.mean(precision)
        ar = np.mean(recall)
        af = np.mean(f1)
        return ap, ar, af

    def prf_at_threshold(
        self, probs: np.ndarray, labels: np.ndarray, thresholds: float | List[float] = None
    ):
        thresholds = self._set_thresholds(thresholds)
        metrics_at_t = []
        for t in thresholds:
            pred_cls = np.where(probs >= t, 1, 0)
            ap, ar, af = self.precision_recall_f1(pred_cls, labels)
            metrics_at_t.append([ap, ar, af])
        return np.mean(metrics_at_t, axis=0)

    def compute(self, input_data: EvalPrediction) -> Dict[str, Any]:
        probs = sigmoid(input_data.predictions)
        labels = input_data.label_ids
        metrics = {}

        p_t, r_t, f_t = self.prf_at_threshold(probs, labels, thresholds=0.5)
        map, mar, maf = self.prf_at_threshold(probs, labels)

        metrics["AP@0.5"] = p_t
        metrics["AR@0.5"] = r_t
        metrics["AF1@0.5"] = f_t

        metrics["mAP"] = map
        metrics["mAR"] = mar
        metrics["mAF1"] = maf

        return metrics
