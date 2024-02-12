import argparse
from pathlib import Path

from tqdm import tqdm

from nonwestlit.evaluate import evaluate


def main(args):
    data_path = Path(args.path).resolve()
    evaluated_outputs_path = data_path / "evaluated_outputs.txt"
    evaluated_outputs_path.touch(exist_ok=True)
    with open(evaluated_outputs_path, "r") as fd_in:
        evaluated_outputs = [p.strip() for p in fd_in.readlines()]
    print(evaluated_outputs)
    for model_dir in tqdm(data_path.glob("*")):
        for checkpoint_path in model_dir.glob("*"):
            if not checkpoint_path.name.startswith("checkpoint-") or checkpoint_path.as_posix() in evaluated_outputs:
                break
            evaluate(model_name_or_path=checkpoint_path, data_path=args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to the model directory.")
    parser.add_argument("data_path", type=str, help="Path to the dataset folder.")
    args = parser.parse_args()
    main(args)


