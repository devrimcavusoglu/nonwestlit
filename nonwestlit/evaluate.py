from pathlib import Path
from typing import Tuple

import numpy as np
import tqdm
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, EvalPrediction, pipeline

from nonwestlit.data_utils import load_hf_data
from nonwestlit.metrics import MultiLabelClassificationMetrics, SingleLabelClassificationMetrics
from nonwestlit.utils import NonwestlitTaskTypes, geometric_mean, sigmoid, softmax, create_neptune_run


def prepare_chunks_for_articles(test_dataset) -> Tuple[list[list[str]], list[int]]:
    articles = []
    labels = []

    current_article = []
    current_label = None
    current_title = None
    for chunk in test_dataset:
        title = chunk["title"]
        chunk_text = chunk["input_ids"]
        if title == current_title:
            current_label = chunk["labels"]
            current_article.append(chunk_text)
        else:
            if current_label is not None:
                articles.append(current_article)
                labels.append(current_label)
            current_article = [chunk_text]
            current_label = chunk["labels"]
            current_title = title
    return articles, labels


def get_pred_scores(preds, task: str) -> np.ndarray:
    scores = []
    for pred in preds:
        pred_scores = [p["score"] for p in pred]
        scores.append(pred_scores)
    all_scores = np.array(scores)
    if task == NonwestlitTaskTypes.seq_cls:
        all_probs = np.apply_along_axis(softmax, axis=1, arr=all_scores)
    else:
        all_probs = np.apply_along_axis(sigmoid, axis=1, arr=all_scores)
    return all_probs.mean(axis=0)


def evaluate(
    model_path: str,
    neptune_project_name: str,
    data_path: str,
    num_labels: int,
    max_sequence_length: int,
    task_type: str = "sequence-classification",
):
    """
    Main evaluation function for the test runs of the fine-tuned models.

    Args:
        model_path (str): Fine-tuned model checkpoint.
        neptune_project_name (str): Neptune project name for logging, name in format of PROJECT_NAME and not in
            WORKSPACE_NAME/PROJECT_NAME, WORKSPACE='nonwestlit' is reserved, prepended and cannot be changed. This
            parameter is set as positional argument to avoid having conflicts.
        num_labels (int): Number of labels in the dataset.
        data_path (str): Path to the test dataset folder.
        task_type (str): Task type of the training. Set as 'sequence-classification' by default.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, load_in_8bit=True
    )
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        function_to_apply="none",
    )

    _, _, test_dataset = load_hf_data(
        data_path=data_path,
        tokenizer=tokenizer,
        splits=["test"],
        max_sequence_length=max_sequence_length,
    )

    if task_type == NonwestlitTaskTypes.seq_cls:
        metrics = SingleLabelClassificationMetrics()
    elif task_type == NonwestlitTaskTypes.multi_seq_cls:
        metrics = MultiLabelClassificationMetrics()
    else:
        raise ValueError("Unknown task type '%s'" % task_type)

    articles, labels = prepare_chunks_for_articles(test_dataset)

    all_predictions = []
    total_len = len(articles)
    for article_chunks, label in tqdm.tqdm(zip(articles, labels), total=total_len):
        predictions = pipe(article_chunks)
        final_probs = get_pred_scores(predictions, task_type)  # Apply avg. pooling over softmax.
        all_predictions.append(final_probs)

    eval_pred = EvalPrediction(predictions=np.array(all_predictions), label_ids=np.array(labels))
    metric_results = metrics(eval_pred, function_to_apply=None)
    print(metric_results)
    run = create_neptune_run(neptune_project_name, experiment_tracking=True)
    run["metrics"] = metric_results
    run["sys/tags"].add("evaluation")
    model_path = Path(model_path).resolve()
    model_name = model_path.parent.name if model_path.name.startswith("checkpoint-") else model_path.name
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


if __name__ == "__main__":
    # An issue related with loading a model checkpoint.
    # https://github.com/huggingface/peft/issues/577

    # single label evaluation
    model_path = "/home/devrim/lab/gh/ms/nonwestlit/outputs/russian_first_level_llama_2_lora_seq_cls_chunks/checkpoint-2640"
    data_path = "/home/devrim/lab/gh/ms/nonwestlit/data/russian_first_level"
    task = "sequence-classification"
    max_seq_len = 2048
    evaluate(neptune_project_name="first-level-classification", model_path=model_path, data_path=data_path, max_sequence_length=max_seq_len, num_labels=3)
