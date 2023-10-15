from dataclasses import dataclass
from typing import List, Dict, Any

from transformers import DataCollator, PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class NONWESTLITDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for instance in features:
            tokens = self.tokenizer(instance["text"], truncation=True)
            batch["input_ids"].append(tokens["input_ids"])
            batch["attention_mask"].append(tokens["attention_mask"])
            batch["labels"].append(instance["text_type"])
        return batch


if __name__ == "__main__":
    from src.dataset import NONWESTLITDataset
    from torch.utils.data import DataLoader

    model_name = "tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = NONWESTLITDataset("/home/devrim/lab/gh/nonwestlit/test_data/toy_dataset.json")
    collator = NONWESTLITDataCollator(tokenizer=tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    for x in train_loader:
        print(x)
        break

