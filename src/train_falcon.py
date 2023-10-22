from typing import List, Optional

import fire
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    FalconForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.collator import NONWESTLITDataCollator
from src.dataset import NONWESTLITDataset


def init_model(model_name_or_path: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )
    return tokenizer, model


def train(
    model_name_or_path: str,
    data_path: str,
    num_labels: int = 3,
    batch_size: int = 2,
    n_epochs: int = 3,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    freeze_base: bool = True,
):
    model: FalconForSequenceClassification
    tokenizer, model = init_model(model_name_or_path, num_labels)
    if freeze_base:
        if "falcon" in model_name_or_path:
            # freeze the base/backbone transformer, `model.transformer` is specific to HF Falcon
            model.transformer.requires_grad_(False)
    dataset = NONWESTLITDataset(data_path)
    collator = NONWESTLITDataCollator(tokenizer=tokenizer)
    use_cpu = device == "cpu"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        use_cpu=use_cpu,
        do_train=True,
        save_strategy="epoch",
        num_train_epochs=n_epochs,
        metric_for_best_model="train_loss",
        greater_is_better=False,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )
    return trainer.train()


def predict(
    model_name_or_path: str,
    inputs: List[str],
    device: Optional[str] = None,
):
    """
    Uses near 37 GB of memory on CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=tokenizer, device=device)
    out = pipe(inputs)
    return out


if __name__ == "__main__":
    fire.Fire({"train": train, "generate": predict})
