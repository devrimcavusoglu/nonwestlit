import warnings
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, PeftConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    FalconForSequenceClassification,
    LlamaForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments, PreTrainedTokenizerBase,
)
from transformers.utils import is_peft_available

from nonwestlit.collator import NonwestlitSequenceClassificationDataCollator, NonwestlitCausalLMDataCollator
from nonwestlit.dataset import NONWESTLITDataset
from nonwestlit.utils import Nullable, print_trainable_parameters, TaskTypes

TASK_TO_LORA = {"causal-lm": "text-generation", "sequence-classification": "SEQ_CLS"}


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


def _freeze_backbone(model: PreTrainedModel) -> None:
    if "falcon" in model.name_or_path:
        #  Without `freeze_backbone` it's nearly impossible to train the 7b vanilla model as it surpasses over
        #  128 GiB memory, with linear probing (freezing the entire backbone except the classification head) the
        #  memory use on CPU is around 30 GiB.
        model: FalconForSequenceClassification
        # freeze the base/backbone transformer, `model.transformer` is specific to `FalconForSequenceClassification`
        model.transformer.requires_grad_(False)
    if "llama" in model.name_or_path:
        model: LlamaForSequenceClassification
        # freeze the base/backbone transformer, `model.model` is specific to `LlamaForSequenceClassification`
        model.model.requires_grad_(False)


def _get_adapter_config(adapter: str, target_modules: List[str], task_type: str) -> PeftConfig:
    if not is_peft_available():
        raise EnvironmentError(
            "Training w/ Lora requires the package `peft`. Install it by `pip install peft`."
        )
    if adapter == "lora":
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            target_modules=target_modules,
            task_type=TASK_TO_LORA[task_type],
        )


def _construct_model(
    model_name_or_path: str, task_type: str, quantization_cfg, num_labels: int
) -> PreTrainedModel:
    if task_type == TaskTypes.seq_cls:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            quantization_config=quantization_cfg,
        )
    elif task_type == TaskTypes.casual_lm:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=quantization_cfg
        )


def init_model(
    model_name_or_path: str,
    num_labels: int,
    bnb_quantization: str,
    adapter: str,
    lora_target_modules: List[str],
    gradient_checkpointing: bool,
    freeze_backbone: bool,
    task_type: str,
) -> Tuple:
    model: PreTrainedModel
    quantization_cfg = setup_bnb_quantization(bnb_quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = _construct_model(model_name_or_path, task_type, quantization_cfg, num_labels)
    if bnb_quantization is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    if adapter is not None:
        peft_cfg = _get_adapter_config(adapter, lora_target_modules, task_type)
        if peft_cfg is not None:
            model.add_adapter(peft_cfg)
    elif freeze_backbone:
        _freeze_backbone(model)

    if tokenizer.pad_token is None:
        # Adding a new PAD token.
        # https://stackoverflow.com/a/73137031
        # pad token can optionally be set to eos token, but we will define a new one.
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def _get_collator(task_type: str, tokenizer: PreTrainedTokenizerBase):
    if task_type == TaskTypes.seq_cls:
        return NonwestlitSequenceClassificationDataCollator(tokenizer=tokenizer)
    elif task_type == TaskTypes.casual_lm:
        return NonwestlitCausalLMDataCollator(tokenizer)


def train(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    freeze_backbone: Optional[bool] = True,
    adapter: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    bnb_quantization: Optional[str] = None,
    gradient_checkpointing: bool = True,
    task_type: str = "sequence-classification",
    **kwargs
):
    """
    Main training function for the training of the base pretrained models.

    Args:
        model_name_or_path (str): Model name from the HF Hub or path.
        data_path (str): Path to the dataset file.
        output_dir (str): Path to a directory where the model directory is saved under.
        freeze_backbone (bool):  If true, backbone transformer is frozen and only head is trained.
            For training adapters, backbone is always frozen.
        adapter (str): Adapter method (e.g. 'lora'). By default `None`.
        lora_target_modules (List(str)): Target module names for the lora adapters to be trained on.
        bnb_quantization (str): BitsAndBytes quantization type.
        gradient_checkpointing (bool): If true, gradient checkpointing is used if applicable.
        task_type (str): Task type of the training. Set as 'sequence-classification' by default.
        kwargs: All keyword arguments for the :py:class:`TrainingArguments`. To see the supported arguments, see
            the documentation below.
            https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.TrainingArguments

    Return:
         (TrainingOutput) Training output metrics of the training.
    """
    if "num_labels" in kwargs:
        num_labels = kwargs.pop("num_labels")
    else:
        num_labels = None
    if task_type == "sequence-classification" and num_labels is None:
        raise TypeError(
            "If `task_type` is 'sequence-classification', then `num_labels` parameter has to be passed."
        )
    elif task_type == "causal-lm" and num_labels is not None:
        warnings.warn(
            "Parameters 'num_labels' is passed, but task_type is 'causal-lm'. 'num_labels' will be igonred."
        )

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
        freeze_backbone=freeze_backbone,
        task_type=task_type,
    )

    print_trainable_parameters(model)
    dataset = NONWESTLITDataset(data_path)
    collator = _get_collator(task_type, tokenizer)
    training_args = TrainingArguments(output_dir=output_dir, do_train=True, **kwargs)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )
    return trainer.train()
