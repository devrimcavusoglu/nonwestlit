import argparse
from pathlib import Path

from nonwestlit.utils import read_json


def correct_nans(path: str):
    path = Path(path)
    for file in path.rglob("*/*.json"):
        content = file.read_text("utf-8")
        content = content.replace('"label": NaN', '"label": null')
        file.write_text(content)


def count_labels(path: str):
    path = Path(path)
    print("#"*60 + " Data Counts " + "#"*60)
    for file in path.rglob("*/train.json"):
        data = read_json(file)
        total_classes = 1
        for instance in data:
            if "label" not in instance:
                break
            label = instance["label"]
            if isinstance(label, str):
                max_label = max([int(l) for l in label.split(",")])
            elif isinstance(label, int):
                max_label = label
            if max_label > total_classes:
                total_classes = max_label
        if total_classes > 1:
            m_train = len(data)
            m_val = len(read_json(file.parent/'val.json'))
            m_test = len(read_json(file.parent/'test.json'))
            m_total = m_train + m_val + m_test
            print(f"{file.parent}\t| Num classes: {total_classes}")
            print(f"\t> Total: {m_total}\n"
                  f"\t> train: {m_train}\n"
                  f"\t> val: {m_val}\n"
                  f"\t> test: {m_test}")
    print("#"*64 + " END " + "#" * 64)


def main(args):
    if args.count:
        count_labels(args.path)
    else:
        correct_nans(args.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the dataset folder.")
    parser.add_argument("--count", action="store_true", help="If given, return the label counts.")
    args = parser.parse_args()
    main(args)
