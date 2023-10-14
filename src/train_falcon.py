from typing import List, Dict

import fire
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from src.dataset import NONWESTLITDataset


def init_model(model_name_or_path: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification(model_name_or_path, num_labels=num_labels)
    return tokenizer, model


def train(data_loader: DataLoader, tokenizer, model: nn.Module):
    # https://www.shecodes.io/athena/92466-how-to-fine-tune-llama-for-text-classification
    optimizer = torch.optim.SGD(lr=3e-5, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10
    for epochs in range(epochs):
        model.train()
        optimizer.zero_grad()


def main(model_name_or_path: str, data_path: str, num_labels: int = 3, batch_size: int = 2):
    # tiiuae/falcon-7b
    tokenizer, model = init_model(model_name_or_path, num_labels)
    dataset = NONWESTLITDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    train(train_loader, tokenizer, model)


if __name__ == "__main__":
    fire.Fire(main)
