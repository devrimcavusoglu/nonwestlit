from typing import Tuple

import numpy as np
import tqdm
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer, EvalPrediction, pipeline

from nonwestlit.data_utils import load_hf_data
from nonwestlit.metrics import MultiLabelClassificationMetrics, SingleLabelClassificationMetrics
from nonwestlit.utils import NonwestlitTaskTypes, geometric_mean, sigmoid, softmax


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
    data_path: str,
    num_labels: int,
    max_sequence_length: int,
    task_type: str = "sequence-classification",
):
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
        metrics = SingleLabelClassificationMetrics(num_labels=num_labels)
    elif task_type == NonwestlitTaskTypes.multi_seq_cls:
        metrics = MultiLabelClassificationMetrics(num_labels=num_labels)
    else:
        raise ValueError("Unknown task type '%s'" % task_type)

    articles, labels = prepare_chunks_for_articles(test_dataset)

    all_predictions = []
    total_len = len(articles)
    i = 0
    for article_chunks, label in tqdm.tqdm(zip(articles, labels), total=total_len):
        if i > 2:
            break
        predictions = pipe(article_chunks)
        final_probs = get_pred_scores(predictions, task_type)  # Apply avg. pooling over softmax.
        all_predictions.append(final_probs)
        i += 1

    eval_pred = EvalPrediction(predictions=np.array(all_predictions), label_ids=np.array(labels[:3]))
    print(np.where(eval_pred.predictions > 0.5, 1, 0))
    print(eval_pred.label_ids)
    print("=" * 60)
    print(metrics(eval_pred, function_to_apply=None))


if __name__ == "__main__":
    # An issue related with loading a model checkpoint.
    # https://github.com/huggingface/peft/issues/577

    # single label evaluation
    model_path = "/home/devrim/lab/gh/ms/nonwestlit/outputs/russian_first_level_llama_2_lora_seq_cls_chunks_lr175e-7/checkpoint-3300"
    data_path = "/home/devrim/lab/gh/ms/nonwestlit/data/russian_first_level"
    task = "sequence-classification"
    max_seq_len = 2048
    evaluate(model_path=model_path, data_path=data_path, max_sequence_length=max_seq_len, num_labels=3)

    # # multi label evaluation
    # model_path = "/home/devrim/lab/gh/ms/nonwestlit/outputs/llama-2_7b_ottoman_cultural_discourse_subject_lora_seq_cls_lr_5e-6/checkpoint-2485"
    # data_path = "/home/devrim/lab/gh/ms/nonwestlit/data/ottoman_second_level:cultural_discourse_subject"
    # task = "multilabel-sequence-classification"
    # max_seq_len = 2048
    # num_labels = 8
    # evaluate(
    #     model_path=model_path,
    #     data_path=data_path,
    #     max_sequence_length=max_seq_len,
    #     num_labels=num_labels,
    #     task_type=task,
    # )
