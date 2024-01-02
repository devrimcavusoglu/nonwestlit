from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from nonwestlit.pipeline import NONWESTLITClassificationPipeline
from nonwestlit.utils import read_json


class ListDataset(Dataset):
    """
    Simple helper wrapper for lists to accommodate with progress bar. See the
    related comment below.
    https://github.com/huggingface/transformers/issues/14789#issuecomment-998639662
    """

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def predict(
    data_path: str,
    model_name_or_path: str,
    num_labels: int,
    in_sample_batch_size: int = 1,
    max_sequence_length: int = 2048,
    return_scores_only: bool = False,
    task_type: str = None,
):
    """
    Main predict function for predictions input texts from the fine-tuned models.

    Args:
        model_name_or_path (str): Fine-tuned model checkpoint.
        data_path (str): Path to the test dataset folder.
        num_labels (int): Number of labels in the dataset.
        in_sample_batch_size (int): Batch size for the chunked data, is not equal to effective batch size of the
            raw inputs. This argument is mainly used to reduce memory overhead due to chunking as after chunking
            the whole batch (chunks) may not fit in memory.
        max_sequence_length (int): Maximum sequence length for the tokenization and chunking, recommended value
            is the same as the fine-tuned model's max sequence length. The usual value is 2048.
        return_scores_only (bool): If true, raw model outputs/logits will be returned.
        task_type (str): Task type of the prediction. By default, set as 'sequence-classification'. Possible
            values are [sequence-classification, multilabel-sequence-classification].
    """
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_name_or_path, load_in_8bit=True, num_labels=num_labels
    )
    if model.config.pad_token_id is None and tokenizer.pad_token != tokenizer.eos_token:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    pipe = NONWESTLITClassificationPipeline(model=model, tokenizer=tokenizer)
    data = read_json(data_path)
    articles = ListDataset([instance["article"] for instance in data])
    pipe_args = {
        "in_sample_batch_size": in_sample_batch_size,
        "max_sequence_length": max_sequence_length,
        "return_scores_only": return_scores_only,
        "task_type": task_type,
    }

    out = []
    for output in tqdm(pipe(articles, **pipe_args), desc="Prediction"):
        out.append(output)
    return out


if __name__ == "__main__":
    # Single label
    # out = predict(
    #     data_path="/home/devrim/lab/gh/ms/nonwestlit/test_data/toy_train.json",
    #     model_name_or_path="/home/devrim/lab/gh/ms/nonwestlit/outputs/russian_first_level_llama_2_lora_seq_cls_chunks/checkpoint-2640",
    #     num_labels=3,
    #     return_scores_only=True,
    # )

    # Multilabel
    out = predict(
        data_path="/home/devrim/lab/gh/ms/nonwestlit/test_data/toy_test_multilabel.json",
        model_name_or_path="/home/devrim/lab/gh/ms/nonwestlit/outputs/falcon-7b_russian_cultural_discourse_type_lora_seq_cls_lr_1.75e-5/checkpoint-3570",
        num_labels=7,
        in_sample_batch_size=1,
        return_scores_only=True,
        task_type="multilabel-sequence-classification",
    )
    print(out)
