import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import LoraConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    FalconForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    pipeline,
)
from transformers.utils import is_peft_available

from nonwestlit.collator import NONWESTLITDataCollator
from nonwestlit.dataset import NONWESTLITDataset
from nonwestlit.utils import Nullable


def setup_bnb_quantization(bnb_quantization: Optional[str] = None) -> Nullable[BitsAndBytesConfig]:
    if bnb_quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif bnb_quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def init_model(
    model_name_or_path: str,
    num_labels: int,
    bnb_quantization: str,
    adapter: str,
    lora_target_modules: List[str],
) -> Tuple:
    model: PreTrainedModel
    quantization_cfg = setup_bnb_quantization(bnb_quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        quantization_config=quantization_cfg,
    )
    if adapter is not None:
        if not is_peft_available():
            raise EnvironmentError(
                "Training w/ Lora requires the package `peft`. Install it by `pip install peft`."
            )
        if adapter == "lora":
            peft_cfg = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                target_modules=lora_target_modules,
                task_type="SEQ_CLS",
            )
            model.add_adapter(peft_cfg)
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
    freeze_backbone: Optional[bool] = True,
    adapter: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    quantization: Optional[str] = None,
    half_precision_backend: Optional[str] = "auto",
    optim: Optional[str] = "adamw_torch",
    optim_args: Optional[Dict[str, Any]] = None,
    deepspeed: Optional[Union[str, Dict]] = None,
):
    if quantization in ["4bit", "8bit"] and device.lower() == "cpu":
        warnings.warn(
            "4 and 8 bit quantization is not supported on CPU, forcefully setting effective device to GPU."
        )
    tokenizer, model = init_model(
        model_name_or_path,
        num_labels,
        bnb_quantization=quantization,
        adapter=adapter,
        lora_target_modules=lora_target_modules,
    )
    if freeze_backbone:
        if "falcon" in model_name_or_path and adapter is None:
            #  Without `freeze_backbone` it's nearly impossible to train the 7b vanilla model as it surpasses over
            #  128 GiB memory, with linear probing (freezing the entire backbone except the classification head) the
            #  memory use on CPU is around 30 GiB.
            model: FalconForSequenceClassification
            # freeze the base/backbone transformer, `model.transformer` is specific to HF Falcon
            model.transformer.requires_grad_(False)
    dataset = NONWESTLITDataset(data_path)
    collator = NONWESTLITDataCollator(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        use_cpu=device == "cpu",
        do_train=True,
        save_strategy="epoch",
        num_train_epochs=n_epochs,
        metric_for_best_model="train_loss",
        greater_is_better=False,
        save_total_limit=1,
        bf16=quantization == "bf16",
        fp16=quantization == "fp16",
        half_precision_backend=half_precision_backend,
        optim=optim,
        optim_args=optim_args,
        deepspeed=deepspeed,
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
    Falcon-7b Uses roughly 32 GiB of memory on CPU, peak is around 36 GiB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=tokenizer, device=device)
    out = pipe(inputs)
    return out
