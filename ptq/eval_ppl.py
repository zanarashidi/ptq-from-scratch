"""WikiText-2 perplexity - non-overlapping sliding window, the standard PTQ metric."""
import torch
from tqdm import tqdm

from .data import get_eval_tokens


@torch.no_grad()
def eval_ppl(model, tokenizer, seqlen: int = 2048, device=None, limit_windows=None):
    device = device or next(model.parameters()).device
    ids = get_eval_tokens(tokenizer, "test").to(device)
    n_windows = ids.shape[1] // seqlen
    if limit_windows:
        n_windows = min(n_windows, limit_windows)

    nll_sum, n_tokens = 0.0, 0
    for i in tqdm(range(n_windows), desc="ppl"):
        batch = ids[:, i * seqlen : (i + 1) * seqlen]
        logits = model(batch).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += shift_labels.numel()

    return torch.exp(torch.tensor(nll_sum / n_tokens)).item()
