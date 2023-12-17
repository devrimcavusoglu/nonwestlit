import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import neptune
import numpy as np
import torch
from neptune import Run
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    FalconForSequenceClassification,
    LlamaForSequenceClassification,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import NeptuneCallback
from transformers.utils import is_peft_available

from nonwestlit import NEPTUNE_CFG
from nonwestlit.data_utils import get_collator, load_hf_data, load_torch_data
from nonwestlit.metrics import MultiLabelClassificationMetrics, SingleLabelClassificationMetrics
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
        "multilabel-sequence-classification": TaskType.SEQ_CLS,
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
    if task_type in [NonwestlitTaskTypes.seq_cls, NonwestlitTaskTypes.multi_seq_cls]:
        problem_type = (
            "multi_label_classification"
            if NonwestlitTaskTypes.multi_seq_cls
            else "single_label_classification"
        )
        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            quantization_config=quantization_cfg,
            problem_type=problem_type,
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


def _check_neptune_creds(neptune_project_name: str):
    if os.getenv("NEPTUNE_PROJECT") is not None and os.getenv("NEPTUNE_API_TOKEN") is not None:
        return
    elif NEPTUNE_CFG.exists():
        cfg = read_cfg(NEPTUNE_CFG.as_posix())
        neptune_project = cfg[neptune_project_name]["project"]
        neptune_token = cfg[neptune_project_name]["api_token"]
        assert neptune_project is not None and neptune_token is not None
        # set as environment variables
        os.environ["NEPTUNE_PROJECT"] = neptune_project
        os.environ["NEPTUNE_API_TOKEN"] = neptune_token
        return
    raise EnvironmentError("Neither environment variables nor neptune.cfg is found.")


def create_neptune_callback(
    neptune_project_name: str, experiment_tracking: bool, callbacks: List
) -> Nullable[Run]:
    if not experiment_tracking:
        return None
    _check_neptune_creds(neptune_project_name)
    run = neptune.init_run()
    neptune_callback = NeptuneCallback(run=run)
    callbacks.append(neptune_callback)
    return run


def train(
    model_name_or_path: str,
    neptune_project_name: str,
    output_dir: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    dataset_framework: Optional[str] = "hf",
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
        neptune_project_name (str): Neptune project name for logging, name in format of PROJECT_NAME and not in
            WORKSPACE_NAME/PROJECT_NAME, WORKSPACE='nonwestlit' is reserved, prepended and cannot be changed. This
            parameter is set as positional argument to avoid having conflicts.
        output_dir (str): Path to a directory where the model directory is saved under.
        train_data_path (str): Path to the training dataset file.
        eval_data_path (Optional(str)): Path to the evaluation dataset file.
        dataset_framework (Optional(str): Framework for dataset to be used. Valid choices: [hf, torch].
        freeze_backbone (Optional(bool)):  If true, backbone transformer is frozen and only head is trained.
            For training adapters, backbone is always frozen.
        adapter (Optional(str)): Adapter method (e.g. 'lora'). By default, `None`.
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
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Given output directory '{output_dir.as_posix()}' exists. This check prevents "
            f"accidental overwriting to the existing directories which may result in loss of model "
            f"weights."
        )
    if (
        task_type in [NonwestlitTaskTypes.seq_cls, NonwestlitTaskTypes.multi_seq_cls]
        and num_labels is None
    ):
        raise TypeError(
            "If `task_type` is 'sequence-classification' or , then `num_labels` parameter has to be passed."
        )
    elif task_type == NonwestlitTaskTypes.casual_lm and num_labels is not None:
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
    max_sequence_length = max_sequence_length or tokenizer.model_max_length

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
    else:
        print_trainable_parameters(model)

    if "evaluation_strategy" not in kwargs and eval_data_path is not None:
        kwargs["evaluation_strategy"] = "steps"

    if task_type == NonwestlitTaskTypes.seq_cls:
        compute_metrics = SingleLabelClassificationMetrics(num_labels=num_labels)
    elif task_type == NonwestlitTaskTypes.multi_seq_cls:
        compute_metrics = MultiLabelClassificationMetrics(num_labels=num_labels)
    else:
        warnings.warn("No evalutor is set up. 'compute_metrics' is set to None.")
        compute_metrics = None

    training_args = TrainingArguments(
        output_dir=output_dir.as_posix(), do_train=True, report_to="none", **kwargs
    )
    if dataset_framework == "torch":
        collator = get_collator(
            task_type, tokenizer, num_virtual_tokens, max_sequence_length, is_mapping=False
        )
        train_dataset, eval_dataset, _ = load_torch_data(train_data_path, eval_data_path)
    elif dataset_framework == "hf":
        assert train_data_path == eval_data_path
        collator = get_collator(
            task_type, tokenizer, num_virtual_tokens, max_sequence_length, is_mapping=True
        )
        train_dataset, eval_dataset, _ = load_hf_data(
            train_data_path, tokenizer, collator, max_sequence_length=max_sequence_length
        )
        collator = None  # No need for collator to be used, we utilized mapping which is faster.
    else:
        raise ValueError(
            f"Only valid choices for 'dataset_framework' argument are [hf,torch], got {dataset_framework}"
        )

    callbacks = []
    run = create_neptune_callback(neptune_project_name, experiment_tracking, callbacks)
    if run is not None:
        aux_data = {
            "train_data": train_data_path,
            "eval_data": eval_data_path,
            "dataset_framework": dataset_framework,
            "freeze_backbone": freeze_backbone,
            "adapter": adapter,
            "lora_target_modules": str(lora_target_modules),
            "bnb_quantization": bnb_quantization,
            "gradient_checkpointing": gradient_checkpointing,
            "task_type": task_type,
            "max_sequence_length": max_sequence_length,
            "task_specific/num_labels": num_labels,
            "task_specific/num_virtual_tokens": num_virtual_tokens,
        }
        run["entrypoint_args"] = aux_data
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    return trainer.train()
