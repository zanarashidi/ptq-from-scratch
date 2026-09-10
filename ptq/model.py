"""Model / device helpers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(name: str = DEFAULT_MODEL, dtype: torch.dtype = torch.float16):
    """Load model on CPU (caller moves to device). eval + no grad by default."""
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def iter_quantizable_linears(model):
    """Yield (name, module) for nn.Linear layers inside transformer blocks.

    Skips the embedding and the lm_head / output projection - those stay in
    fp16 in every serious PTQ setup and quantizing them tanks perplexity.
    """
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "lm_head" in name or name.endswith("embed_tokens"):
            continue
        yield name, module
