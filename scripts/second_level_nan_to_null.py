import argparse
from pathlib import Path


def correct_nans(path: str):
    path = Path(path)
    for file in path.rglob("*/*.json"):
        content = file.read_text("utf-8")
        content = content.replace('"label": NaN', '"label": null')
        file.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the dataset folder.")
    args = parser.parse_args()
    correct_nans(args.path)
