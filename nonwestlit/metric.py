from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, Any

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

    def compute(self, input_data: EvalPrediction) -> Dict[str, Any]:
        pred_cls = np.where(sigmoid(input_data.predictions) > self.threshold, 1, 0)
        labels = input_data.label_ids
        metrics = {}
        n = np.prod(pred_cls.shape)
        hits = pred_cls == labels
        pred_support = np.sum(pred_cls, axis=0)
        gt_support = np.sum(labels, axis=0)

        metrics["val_accuracy"] = np.sum(hits) / n

        # AP (Average Precision)
        p = np.sum(np.logical_and(pred_cls == 1, hits), axis=0) / pred_support
        p = np.nan_to_num(p)
        metrics["avg_precision"] = np.mean(p)

        # AR (Average Recall)
        r = np.sum(np.logical_and(labels == 1, hits), axis=0) / gt_support
        r = np.nan_to_num(r)
        metrics["avg_recall"] = np.mean(r)

        return metrics


if __name__ == "__main__":
    np.random.seed(42)
    a = np.array([[1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1, 0, 0]], dtype=np.uint8)
    b = np.random.randn(2, 8)
    p = EvalPrediction(label_ids=a, predictions=b)
    evaluator = MultiLabelClassificationEvaluator(num_labels=8)
    m = evaluator(p)
    print(m)