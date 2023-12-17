from typing import List

from transformers import AutoTokenizer, pipeline


def predict(
    model_name_or_path: str,
    inputs: List[str],
):
    """
    Falcon-7b Uses roughly 32 GiB of memory on CPU, peak is around 36 GiB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=tokenizer, model_kwargs={"load_in_8bit": True})
    out = pipe(inputs)
    return out
