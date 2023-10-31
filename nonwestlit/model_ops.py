import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
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
from nonwestlit.utils import Nullable, print_trainable_parameters


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
    gradient_checkpointing: bool,
) -> Tuple:
    model: PreTrainedModel
    quantization_cfg = setup_bnb_quantization(bnb_quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        quantization_config=quantization_cfg,
    )
    if bnb_quantization is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
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
    freeze_backbone: Optional[bool] = True,
    adapter: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    bnb_quantization: Optional[str] = None,
    gradient_checkpointing: bool = True,
    *,
    task_type: str = "sequence-classification",
    num_labels: int = None,
    **kwargs
):
    """
    Main training function for the training of the base pretrained models.

    Args:
        model_name_or_path (str): Model name from the HF Hub or path.
        data_path (str): Path to the dataset file.
        freeze_backbone (bool):  If true, backbone transformer is frozen and only head is trained.
            For training adapters, backbone is always frozen.
        adapter (str): Adapter method (e.g. 'lora'). By default `None`.
        lora_target_modules (List(str)): Target module names for the lora adapters to be trained on.
        bnb_quantization (str): BitsAndBytes quantization type.
        gradient_checkpointing (bool): If true, gradient checkpointing is used if applicable.
        task_type (str): Task type of the training. Set as 'sequence-classification' by default.
        kwargs: All keyword arguments for the :py:class:`TrainingArguments`.

    Return:
         (TrainingOutput) Training output metrics of the training.
    """
    if task_type == "sequence-classification" and num_labels is None:
        raise TypeError("If `task_type` is 'sequence-classification', then `num_labels` parameter has to be passed.")
    if bnb_quantization in ["4bit", "8bit"] and kwargs.get("use_cpu", False):
        warnings.warn(
            "4 and 8 bit quantization is not supported on CPU, forcefully setting the effective device to GPU."
        )
    tokenizer, model = init_model(
        model_name_or_path,
        num_labels,
        bnb_quantization=bnb_quantization,
        adapter=adapter,
        lora_target_modules=lora_target_modules,
        gradient_checkpointing=gradient_checkpointing,
    )
    if freeze_backbone:
        if "falcon" in model_name_or_path and adapter is None:
            #  Without `freeze_backbone` it's nearly impossible to train the 7b vanilla model as it surpasses over
            #  128 GiB memory, with linear probing (freezing the entire backbone except the classification head) the
            #  memory use on CPU is around 30 GiB.
            model: FalconForSequenceClassification
            # freeze the base/backbone transformer, `model.transformer` is specific to HF Falcon
            model.transformer.requires_grad_(False)

    print_trainable_parameters(model)
    dataset = NONWESTLITDataset(data_path)
    collator = NONWESTLITDataCollator(tokenizer=tokenizer)
    training_args = TrainingArguments(
        do_train=True,
        **kwargs
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
