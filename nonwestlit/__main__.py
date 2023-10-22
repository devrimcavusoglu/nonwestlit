import fire

from nonwestlit.model_ops import train, predict

if __name__ == "__main__":
    fire.Fire({"train": train, "generate": predict})
