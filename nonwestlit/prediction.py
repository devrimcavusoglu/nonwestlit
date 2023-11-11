from typing import List, Optional

from transformers import AutoTokenizer, pipeline


def predict(
    model_name_or_path: str,
    inputs: List[str],
    device: Optional[str] = None,
):
    """
    Falcon-7b Uses roughly 32 GiB of memory on CPU, peak is around 36 GiB.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-classification", model=model_name_or_path, tokenizer=tokenizer, device=device)
    out = pipe(inputs)
    return out
