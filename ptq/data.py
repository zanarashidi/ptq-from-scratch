"""WikiText-2 calibration and evaluation data."""
import torch
from datasets import load_dataset


def _wikitext2_text(split: str) -> str:
    # Salesforce mirror is parquet-native (no dataset script -> no hub-version issues)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def get_eval_tokens(tokenizer, split: str = "test") -> torch.Tensor:
    """Full concatenated split as a single [1, N] token tensor."""
    enc = tokenizer(_wikitext2_text(split), return_tensors="pt")
    return enc.input_ids


def get_calibration(tokenizer, nsamples: int = 128, seqlen: int = 512, seed: int = 0):
    """Return a list of [1, seqlen] token windows sampled from the train split."""
    ids = tokenizer(_wikitext2_text("train"), return_tensors="pt").input_ids
    g = torch.Generator().manual_seed(seed)
    n = ids.shape[1]
    samples = []
    for _ in range(nsamples):
        start = torch.randint(0, n - seqlen - 1, (1,), generator=g).item()
        samples.append(ids[:, start : start + seqlen])
    return samples
