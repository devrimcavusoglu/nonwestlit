from pathlib import Path

import numpy as np
from transformers import EvalPrediction

from nonwestlit.data_utils import _get_label
from nonwestlit.metrics import MultiLabelClassificationMetrics, SingleLabelClassificationMetrics
from nonwestlit.prediction import predict
from nonwestlit.utils import NonwestlitTaskTypes, create_neptune_run, read_json


def evaluate(
    model_path: str,
    data_path: str,
    num_labels: int,
    max_sequence_length: int,
    task_type: str = "sequence-classification",
    neptune_project_name: str | None = None,
):
    """
    Main evaluation function for the test runs of the fine-tuned models.

    Args:
        model_path (str): Fine-tuned model checkpoint.
        data_path (str): Path to the test dataset folder.
        num_labels (int): Number of labels in the dataset.
        task_type (str): Task type of the training. Set as 'sequence-classification' by default.
        neptune_project_name (str): Neptune project name for logging, name in format of PROJECT_NAME and not in
            WORKSPACE_NAME/PROJECT_NAME, WORKSPACE='nonwestlit' is reserved, prepended and cannot be changed. This
            parameter is set as positional argument to avoid having conflicts.
    """
    if task_type == NonwestlitTaskTypes.seq_cls:
        multi_label = False
        metrics = SingleLabelClassificationMetrics()
    elif task_type == NonwestlitTaskTypes.multi_seq_cls:
        multi_label = True
        metrics = MultiLabelClassificationMetrics()
    else:
        raise ValueError(
            "Unknown task type '%s'. Has to be on of "
            "[sequence-classification, multilabel-sequence-classification]" % task_type
        )

    data = read_json(data_path)
    labels = [_get_label(article, multi_label=multi_label, num_labels=num_labels) for article in data]
    all_predictions = predict(data_path=data_path, model_name_or_path=model_path, num_labels=num_labels, return_scores_only=True)

    eval_pred = EvalPrediction(predictions=np.array(all_predictions), label_ids=np.array(labels))
    metric_results = metrics(eval_pred, function_to_apply=None)
    print(metric_results)
    if neptune_project_name is not None:
        run = create_neptune_run(neptune_project_name)
        run["metrics"] = metric_results
        run["sys/tags"].add("evaluation")
        model_path = Path(model_path).resolve()
        model_name = (
            model_path.parent.name if model_path.name.startswith("checkpoint-") else model_path.name
        )
        data_path = Path(data_path).resolve()
        aux_data = {
            "model_name": model_name,
            "eval_data": f"{data_path.parent.name}/{data_path.name}",
            "task_type": task_type,
            "max_sequence_length": max_sequence_length,
            "task_specific/num_labels": num_labels,
        }
        run["entrypoint_args"] = aux_data
        run.wait()
