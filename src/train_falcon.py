from typing import Optional, Any, Dict, List

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
    if tokenizer.pad_token is None:
        # Adding a new PAD token.
        # https://stackoverflow.com/a/73137031
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
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
    bf16: Optional[bool] = False,
    fp16: Optional[bool] = False,
    half_precision_backend: Optional[str] = "auto",
    optim: Optional[str] = "adamw_torch",
    optim_args: Optional[Dict[str, Any]] = None,
):
    model: FalconForSequenceClassification
    tokenizer, model = init_model(model_name_or_path, num_labels)
    if freeze_base:
        #  Without `freeze_base` it's nearly impossible to train the 7b vanilla model as it surpasses over 128 GiB
        #  memory, with linear probing (freezing the entire backbone except the classification head) the memory use on
        #  CPU is around 30 GiB.
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
        bf16=bf16,
        fp16=fp16,
        half_precision_backend=half_precision_backend,
        optim=optim,
        optim_args=optim_args
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
    Uses roughly 32 GiB of memory on CPU, peak is around 36 GiB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=tokenizer, device=device)
    out = pipe(inputs)
    return out


if __name__ == "__main__":
    fire.Fire({"train": train, "generate": predict})
