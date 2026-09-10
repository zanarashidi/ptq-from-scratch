"""Memory + throughput, framed the way ElastixAI frames it: bytes moved per token.

Note: this repo does fake quantization, so runtime memory here reflects fp16
tensors. The `theoretical_bytes_per_token` number is what matters for the writeup
- it's the weight bytes that must be streamed from HBM per forward pass, which is
the actual bottleneck for memory-bound decode.
"""
import time

import torch

from .model import iter_quantizable_linears


def theoretical_weight_bytes(model, bits: int) -> dict:
    """Bytes to store all block-linear weights at `bits`, vs fp16."""
    quant_bits, fp16_bits = 0, 0
    for _, m in iter_quantizable_linears(model):
        numel = m.weight.numel()
        quant_bits += numel * bits
        fp16_bits += numel * 16
    return {
        "fp16_MB": fp16_bits / 8 / 1e6,
        f"int{bits}_MB": quant_bits / 8 / 1e6,
        "compression": fp16_bits / quant_bits,
    }


@torch.no_grad()
def throughput(model, tokenizer, device, prompt_len: int = 64, gen_len: int = 128):
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
    model.config.use_cache = True
    # warmup
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.mps.synchronize() if device.type == "mps" else None

    t0 = time.time()
    out = model.generate(ids, max_new_tokens=gen_len, do_sample=False)
    torch.mps.synchronize() if device.type == "mps" else None
    dt = time.time() - t0

    model.config.use_cache = False
    n_new = out.shape[1] - prompt_len
    return {"tokens_per_s": n_new / dt, "seconds": dt, "new_tokens": n_new}
