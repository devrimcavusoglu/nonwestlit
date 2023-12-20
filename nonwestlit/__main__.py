import fire

from nonwestlit.evaluate import evaluate
from nonwestlit.prediction import predict
from nonwestlit.training import train

if __name__ == "__main__":
    fire.Fire({"train": train, "predict": predict, "evaluate": evaluate})
