from collections import Counter
from typing import Dict, Any

import numpy as np
from transformers import EvalPrediction


def data_evaluation(input_data: EvalPrediction, num_labels: int) -> Dict[str, Any]:
    pred_cls = input_data.predictions.argmax(-1)
    metrics = {}
    metrics["val_accuracy"] = sum(pred_cls == input_data.label_ids) / len(pred_cls)
    for i in range(num_labels):
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


def multi_label_data_evaluation(input_data: EvalPrediction, num_labels: int, threshold: float = 0.5) -> Dict[str, Any]:
    pred_cls = np.where(input_data.predictions.sigmoid() > threshold, 1, 0)
    metrics = {}
    if pred_cls.ndim == 1:
        metrics["avg_accuracy"] = sum(pred_cls == input_data.label_ids.numpy()) / pred_cls.shape[0]
    elif pred_cls.ndim == 2:
        metrics["avg_accuracy"] = sum(sum(pred_cls == input_data.label_ids.numpy())) / pred_cls.shape[0] / pred_cls.shape[1]
    return metrics


if __name__ == "__main__":
    import torch
    a = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float16)
    b = torch.tensor([-516.5, -264.68, 0.7, 14.1, 25.5, 17.7, 4.5, 1.2], dtype=torch.float16)
    p = EvalPrediction(label_ids=a, predictions=b)
    m = multi_label_data_evaluation(p, num_labels=8)
    print(m)