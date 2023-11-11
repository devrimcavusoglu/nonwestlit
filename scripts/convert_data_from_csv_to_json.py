from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def create_args():
    parser = ArgumentParser(prog="data converter", description="CSV to JSON converter")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing dataset files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    return parser.parse_args()


def main(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    for split in input_path.glob("*.csv"):
        data = pd.read_csv(split, encoding="utf-8")
        data = data[~data["label"].isnull()]
        data["id"] = list(range(1, len(data) + 1))
        data["text_type"] = data["label"].astype(int)
        data.drop("label", axis=1, inplace=True)
        with open(output_path / f"{split.stem}.json", 'w', encoding='utf-8') as file:
            data.to_json(file, orient="records", force_ascii=False)


if __name__ == "__main__":
    args = create_args()
    main(args)
