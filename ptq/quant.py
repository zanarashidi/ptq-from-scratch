"""Core quantization primitives - fake (simulated) quantization only.

Everything here returns a de-quantized tensor of the original dtype so we can
measure perplexity without needing packed-int kernels. Packing / real int4
storage is a separate concern (see benchmark.py).
"""
import torch


def _qparams(w: torch.Tensor, bits: int, sym: bool, groupsize: int):
    """Compute per-(out-channel or group) scale and zero-point.

    w: [out, in]. Grouping is along the input dimension.
    Returns scale, zero, each broadcastable against w.
    """
    out_f, in_f = w.shape
    if groupsize and groupsize > 0:
        assert in_f % groupsize == 0, f"{in_f} not divisible by groupsize {groupsize}"
        wg = w.reshape(out_f, in_f // groupsize, groupsize)
        reduce_dim = -1
    else:
        wg = w
        reduce_dim = -1  # per output channel

    if sym:
        qmax = 2 ** (bits - 1) - 1
        amax = wg.abs().amax(dim=reduce_dim, keepdim=True).clamp_(min=1e-8)
        scale = amax / qmax
        zero = torch.zeros_like(scale)
    else:
        qmax = 2**bits - 1
        wmin = wg.amin(dim=reduce_dim, keepdim=True)
        wmax = wg.amax(dim=reduce_dim, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_(min=1e-8)
        zero = torch.round(-wmin / scale)
    return scale, zero, qmax


def quantize_weight(w: torch.Tensor, bits: int, sym: bool = True, groupsize: int = -1):
    """Round-to-nearest fake quantization of a weight matrix [out, in].

    Returns (w_dequant, meta) where meta carries scale/zero for reuse.
    """
    orig_dtype = w.dtype
    w = w.float()
    out_f, in_f = w.shape
    scale, zero, qmax = _qparams(w, bits, sym, groupsize)

    if groupsize and groupsize > 0:
        wg = w.reshape(out_f, in_f // groupsize, groupsize)
        q = torch.clamp(torch.round(wg / scale) + zero, 0 if not sym else -qmax - 1, qmax)
        dq = (q - zero) * scale
        dq = dq.reshape(out_f, in_f)
    else:
        lo = -qmax - 1 if sym else 0
        q = torch.clamp(torch.round(w / scale) + zero, lo, qmax)
        dq = (q - zero) * scale

    return dq.to(orig_dtype), {"scale": scale, "zero": zero, "bits": bits, "sym": sym}


def quantize_scalar(x: torch.Tensor, bits: int, scale: torch.Tensor, zero: torch.Tensor,
                    sym: bool = True) -> torch.Tensor:
    """Quantize+dequantize with *given* params. Used inside the GPTQ column loop."""
    qmax = 2 ** (bits - 1) - 1 if sym else 2**bits - 1
    lo = -qmax - 1 if sym else 0
    q = torch.clamp(torch.round(x / scale) + zero, lo, qmax)
    return (q - zero) * scale
