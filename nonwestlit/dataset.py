from typing import Dict

from torch.utils.data import Dataset

from nonwestlit.utils import read_json


class NONWESTLITDataset(Dataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self._dataset = read_json(data_path)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            start = subscript.start or 0
            stop = subscript.stop or len(self)
            step = subscript.step or 1
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(subscript, list):
            return [self[i] for i in subscript]
        elif not isinstance(subscript, int):
            raise TypeError(f"Expected type of int, got {type(subscript)}.")
        return self.get_item(subscript)

    def __len__(self):
        return len(self._dataset)

    def get_item(self, index: int) -> Dict:
        instance = self._dataset[index]
        return dict(
            iid=instance["id"],
            title=instance["title"],
            input_ids=instance["article"],
            labels=int(instance["text_type"]) - 1,  # text_type index start from 1
        )