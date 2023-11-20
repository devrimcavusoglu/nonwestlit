import os
import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    FalconForSequenceClassification,
    LlamaForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments, EvalPrediction,
)
from transformers.utils import is_peft_available

from nonwestlit import NEPTUNE_CFG
from nonwestlit.collator import (
    NonwestlitCausalLMDataCollator,
    NonwestlitPromptTuningDataCollator,
    NonwestlitSequenceClassificationDataCollator,
)
from nonwestlit.dataset import NONWESTLITDataset
from nonwestlit.utils import NonwestlitTaskTypes, Nullable, print_trainable_parameters, read_cfg

if is_peft_available():
    from peft import (
        LoraConfig,
        PeftConfig,
        PeftModel,
        PromptTuningConfig,
        PromptTuningInit,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    TASK_TO_LORA = {
        "causal-lm": TaskType.CAUSAL_LM,
        "sequence-classification": TaskType.SEQ_CLS,
        "prompt-tuning": TaskType.CAUSAL_LM,
    }


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


def _get_adapter_config(
    adapter: str, target_modules: List[str], task_type: str, model_name_or_path: Optional[str] = None
) -> PeftConfig:
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
    elif adapter == "prompt-tuning":
        return PromptTuningConfig(
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="Classify if the article belong to a type of literary text, cultural discourse or other:",
            tokenizer_name_or_path=model_name_or_path,
            num_virtual_tokens=8,
            task_type=TASK_TO_LORA[task_type],
        )


def _construct_model(
    model_name_or_path: str, task_type: str, quantization_cfg, num_labels: int
) -> PreTrainedModel:
    if task_type == NonwestlitTaskTypes.seq_cls:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            quantization_config=quantization_cfg,
        )
    elif task_type in [NonwestlitTaskTypes.casual_lm, NonwestlitTaskTypes.prompt_tuning]:
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
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    model = _construct_model(model_name_or_path, task_type, quantization_cfg, num_labels)
    if bnb_quantization is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    if adapter is not None:
        peft_cfg = _get_adapter_config(adapter, lora_target_modules, task_type, model_name_or_path)
        model: PeftModel = get_peft_model(model, peft_cfg)
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


def _get_collator(task_type: str, tokenizer: PreTrainedTokenizerBase, num_virtual_tokens: int, max_sequence_length: int):
    if task_type == NonwestlitTaskTypes.seq_cls:
        return NonwestlitSequenceClassificationDataCollator(tokenizer, max_sequence_length)
    elif task_type == NonwestlitTaskTypes.casual_lm:
        return NonwestlitCausalLMDataCollator(tokenizer, max_sequence_length)
    elif task_type == NonwestlitTaskTypes.prompt_tuning:
        return NonwestlitPromptTuningDataCollator(tokenizer, num_virtual_tokens, max_sequence_length)


def _check_neptune_creds():
    if os.getenv("NEPTUNE_PROJECT") is not None and os.getenv("NEPTUNE_API_TOKEN") is not None:
        return
    elif NEPTUNE_CFG.exists():
        cfg = read_cfg(NEPTUNE_CFG.as_posix())
        neptune_project = cfg["credentials"]["project"]
        neptune_token = cfg["credentials"]["api_token"]
        assert neptune_project is not None and neptune_token is not None
        # set as environment variables
        os.environ["NEPTUNE_PROJECT"] = neptune_project
        os.environ["NEPTUNE_API_TOKEN"] = neptune_token
        return
    raise EnvironmentError("Neither environment variables nor neptune.cfg is found.")


def data_evaluation(input_data: EvalPrediction) -> Dict[str, Any]:
    pred_cls = input_data.predictions.argmax(-1)
    metrics = {}
    metrics["val_accuracy"] = sum(pred_cls == input_data.label_ids) / len(pred_cls)
    for i in range(3):  # num_classes
        hits = sum(np.where(pred_cls == i, 1, 0) == np.where(input_data.label_ids == i, 1, -1))
        if sum(pred_cls == i) != 0:
            metrics[f"val_precision_{i}"] = hits / sum(pred_cls == i)
        else:
            metrics[f"val_precision_{i}"] = 0
        if sum(input_data.label_ids == i) != 0:
            metrics[f"val_recall_{i}"] = hits / sum(input_data.label_ids == i)
        else:
            metrics[f"val_recall_{i}"] = 0
        if metrics[f"val_precision_{i}"] + metrics[f"val_recall_{i}"] != 0:
            metrics[f"val_f1_{i}"] = 2 * metrics[f"val_precision_{i}"] * metrics[f"val_recall_{i}"] / (metrics[f"val_precision_{i}"] + metrics[f"val_recall_{i}"])
        else:
            metrics[f"val_f1_{i}"] = 0
    metrics["val_f1_macro"] = (metrics["val_f1_0"] + metrics["val_f1_1"] + metrics["val_f1_2"]) / 3
    return metrics


def train(
    model_name_or_path: str,
    output_dir: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    freeze_backbone: Optional[bool] = True,
    adapter: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    bnb_quantization: Optional[str] = None,
    gradient_checkpointing: bool = True,
    task_type: str = "sequence-classification",
    experiment_tracking: bool = True,
    max_sequence_length: Optional[int] = None,
    **kwargs,
):
    """
    Main training function for the training of the base pretrained models.

    Args:
        model_name_or_path (str): Model name from the HF Hub or path.
        output_dir (str): Path to a directory where the model directory is saved under.
        train_data_path (str): Path to the training dataset file.
        eval_data_path (Optional(str)): Path to the evaluation dataset file.
        freeze_backbone (Optional(bool)):  If true, backbone transformer is frozen and only head is trained.
            For training adapters, backbone is always frozen.
        adapter (Optional(str)): Adapter method (e.g. 'lora'). By default `None`.
        lora_target_modules (Optional(List(str))): Target module names for the lora adapters to be trained on.
        bnb_quantization (Optional(str)): BitsAndBytes quantization type.
        gradient_checkpointing (bool): If true, gradient checkpointing is used if applicable.
        task_type (str): Task type of the training. Set as 'sequence-classification' by default.
        experiment_tracking (bool): If true, the experiment logs are reported to Neptune.
        max_sequence_length (int): Maximum sequence length for input tokens to be truncated. If None, truncates to the
            longest input seq by default.
        kwargs: All keyword arguments for the :py:class:`TrainingArguments`. To see the supported arguments, see
            the documentation below.
            https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/trainer#transformers.TrainingArguments

    Return:
         (TrainingOutput) Training output metrics of the training.
    """
    num_labels = kwargs.pop("num_labels") if "num_labels" in kwargs else None
    num_virtual_tokens = kwargs.pop("num_virtual_tokens") if "num_virtual_tokens" in kwargs else None
    if experiment_tracking:
        _check_neptune_creds()
        report_to = "neptune"
    else:
        report_to = "none"
    if task_type == "sequence-classification" and num_labels is None:
        raise TypeError(
            "If `task_type` is 'sequence-classification', then `num_labels` parameter has to be passed."
        )
    elif task_type == "causal-lm" and num_labels is not None:
        warnings.warn(
            "Parameters 'num_labels' is passed, but task_type is 'causal-lm'. 'num_labels' will be igonred."
        )

    if adapter == NonwestlitTaskTypes.prompt_tuning and num_virtual_tokens is None:
        raise ValueError(
            "If the adapter/task is prompt tuning, parameter `num_virtual_tokens` has to be passed."
        )
    elif (
        adapter == NonwestlitTaskTypes.prompt_tuning and task_type != NonwestlitTaskTypes.prompt_tuning
    ):
        raise ValueError(
            f"Adapter is set to prompt-tuning, but got `task_type={task_type}`. For prompt-tuning, both "
            f"adapter and the task_type must be {adapter}."
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

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
    else:
        print_trainable_parameters(model)
    train_dataset = NONWESTLITDataset(train_data_path)
    eval_dataset = NONWESTLITDataset(eval_data_path) if eval_data_path is not None else None
    if "evaluation_strategy" not in kwargs and eval_data_path is not None:
        kwargs["evaluation_strategy"] = "steps"
    compute_metrics = data_evaluation if task_type == NonwestlitTaskTypes.seq_cls else None
    collator = _get_collator(task_type, tokenizer, num_virtual_tokens, max_sequence_length)
    training_args = TrainingArguments(
        output_dir=output_dir, do_train=True, report_to=report_to, **kwargs
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    return trainer.train()
