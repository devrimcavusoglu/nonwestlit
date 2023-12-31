from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

from nonwestlit.pipeline import NONWESTLITClassificationPipeline
from nonwestlit.utils import read_json


def predict(
    data_path: str, model_name_or_path: str, num_labels: int, task_type: str = "sequence-classification",
return_scores_only: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_name_or_path, load_in_8bit=True, num_labels=num_labels
    )
    if model.config.pad_token_id is None:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    pipe = NONWESTLITClassificationPipeline(model=model, tokenizer=tokenizer, task_type=task_type, return_scores_only=return_scores_only)
    data = read_json(data_path)
    articles = [instance["article"] for instance in data]
    return pipe(articles)


if __name__ == "__main__":
    out = predict(
        data_path="/home/devrim/lab/gh/ms/nonwestlit/test_data/toy_train.json",
        model_name_or_path="/home/devrim/lab/gh/ms/nonwestlit/outputs/russian_first_level_llama_2_lora_seq_cls_chunks/checkpoint-2640",
        num_labels=3,
        return_scores_only=True
    )
    print(out)

