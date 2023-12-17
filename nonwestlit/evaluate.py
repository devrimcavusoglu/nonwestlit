from typing import Tuple

import numpy as np
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForSequenceClassification

from nonwestlit.data_utils import load_hf_data


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


def get_pred_scores(preds) -> np.ndarray:
    scores = []
    for pred in preds:
        pred_scores = [p["score"] for p in pred]
        scores.append(pred_scores)
    return np.array(scores)


def evaluate(model_path: str, data_path: str, num_labels: int, max_sequence_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_path, num_labels=3, load_in_8bit=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, function_to_apply="none")

    _, _, test_dataset = load_hf_data(
        data_path=data_path,
        tokenizer=tokenizer,
        splits=["test"],
        max_sequence_length=max_sequence_length,
    )

    articles, labels = prepare_chunks_for_articles(test_dataset)

    for article_chunks, label in zip(articles, labels):
        if len(article_chunks) > 1:
            preds = pipe(article_chunks)
            preds = get_pred_scores(preds)
            print(preds, label)
            print(preds.argmax(-1))
            break


if __name__ == "__main__":
    # An issue related with loading a model checkpoint.
    # https://github.com/huggingface/peft/issues/577

    model_path = "/home/devrim/lab/gh/ms/nonwestlit/outputs/russian_first_level_llama_2_lora_seq_cls_chunks_lr175e-7/checkpoint-3300"
    data_path = "/home/devrim/lab/gh/ms/nonwestlit/data/russian_first_level"
    task = "sequence-classification"
    max_seq_len = 2048
    evaluate(model_path=model_path, data_path=data_path, max_sequence_length=max_seq_len, num_labels=3)
