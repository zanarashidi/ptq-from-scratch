"""Round-to-nearest baseline: quantize every linear weight independently."""
import torch

from .model import iter_quantizable_linears
from .quant import quantize_weight


@torch.no_grad()
def quantize_model_rtn(model, bits: int, sym: bool = True, groupsize: int = -1):
    """In-place RTN fake-quantization of all block linear layers. The control group."""
    n = 0
    for name, module in iter_quantizable_linears(model):
        dq, _ = quantize_weight(module.weight.data, bits, sym=sym, groupsize=groupsize)
        module.weight.data.copy_(dq)
        n += 1
    print(f"[rtn] quantized {n} linear layers to {bits}-bit "
          f"(sym={sym}, groupsize={groupsize})")
    return model
