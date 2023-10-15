from typing import List, Dict, Optional

import fire
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, FalconForSequenceClassification
import torch

from src.dataset import NONWESTLITDataset


def init_model(model_name_or_path: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    return tokenizer, model


def train(data_loader: DataLoader, tokenizer, model: nn.Module):
    # https://www.shecodes.io/athena/92466-how-to-fine-tune-llama-for-text-classification
    optimizer = torch.optim.SGD(lr=3e-5, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10
    for epochs in range(epochs):
        model.train()
        optimizer.zero_grad()


def main(model_name_or_path: str, data_path: str, num_labels: int = 3, batch_size: int = 2, device: Optional[str] = None):
    model: FalconForSequenceClassification
    tokenizer, model = init_model(model_name_or_path, num_labels)
    model.transformer.requires_grad_(False)  # freeze the base transformer
    dataset = NONWESTLITDataset(data_path)
    use_cpu = True if device == "cpu" else False
    training_args = TrainingArguments(output_dir="/home/devrim/lab/gh/nonwestlit/outputs", per_device_train_batch_size=batch_size, use_cpu=use_cpu, fp16=True)
    trainer = Trainer(model=model, train_dataset=dataset, tokenizer=tokenizer, args=training_args)
    trainer.train()


if __name__ == "__main__":
    # fire.Fire(main)
    model_name = "tiiuae/falcon-7b"
    data_path = "/home/devrim/lab/gh/nonwestlit/test_data/toy_dataset.json"
    main(model_name_or_path=model_name, data_path=data_path, num_labels=3, batch_size=2, device="cpu")
