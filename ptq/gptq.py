"""GPTQ from scratch - Frantar et al. 2023, "GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers".

Implements the layer-wise Optimal Brain Quantization update: quantize the weight
matrix one column at a time and, after each column, apply a closed-form update to
the *remaining* (not-yet-quantized) columns that compensates for the rounding
error, weighted by the inverse Hessian of the layer's reconstruction loss.

  H      = 2/N * sum_n  x_n x_n^T        (X = layer inputs from calibration data)
  update = -(w_q - quant(w_q)) / [H^-1]_qq  *  [H^-1]_q,:

All the linear algebra (Cholesky, inverse) runs in float64. On MPS that has to
be CPU (MPS has no float64 / unreliable linalg); on CUDA we keep it on the GPU,
which is ~5x faster for the Hessian accumulation with large calibration sets.

The driver quantizes one transformer block at a time so only one block's Hessians
live in memory - important on an 8GB machine.
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from .quant import _qparams, quantize_scalar


class GPTQ:
    """Accumulates the Hessian for one nn.Linear and quantizes it."""

    def __init__(self, layer: nn.Linear, linalg_device="cpu"):
        self.layer = layer
        self.ld = torch.device(linalg_device)
        self.rows, self.cols = layer.weight.shape  # [out, in]
        self.H = torch.zeros((self.cols, self.cols), dtype=torch.float64, device=self.ld)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor):
        """inp: [..., in_features] activations feeding this layer."""
        x = inp.reshape(-1, self.cols).to(self.ld, torch.float64)
        n = x.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        x = x * (2.0 / self.nsamples) ** 0.5
        self.H += x.t() @ x

    @torch.no_grad()
    def quantize(self, bits: int, sym: bool = True, groupsize: int = -1,
                 blocksize: int = 128, percdamp: float = 0.01, actorder: bool = False):
        W = self.layer.weight.data.detach().to(self.ld, torch.float64)
        H = self.H.clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0

        perm = inv_perm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            inv_perm = torch.argsort(perm)

        # static per-channel / per-group RTN params from the (permuted) weights
        if groupsize and groupsize > 0:
            scale_full, zero_full, _ = _qparams(W, bits, sym, groupsize)  # [out, ng, 1]
        else:
            scale_full, zero_full, _ = _qparams(W, bits, sym, 0)          # [out, 1]

        damp = percdamp * torch.mean(torch.diag(H))
        H[range(self.cols), range(self.cols)] += damp

        # Hinv as an upper-triangular Cholesky factor of H^-1 (GPTQ Algorithm 2)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        for i1 in range(0, self.cols, blocksize):
            i2 = min(i1 + blocksize, self.cols)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize and groupsize > 0:
                    g = col // groupsize
                    s, z = scale_full[:, g, 0], zero_full[:, g, 0]
                else:
                    s, z = scale_full[:, 0], zero_full[:, 0]

                q = quantize_scalar(w, bits, s, z, sym=sym)
                Q1[:, i] = q
                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        if actorder:
            Q = Q[:, inv_perm]

        self.layer.weight.data.copy_(Q.to(self.layer.weight.dtype).to(self.layer.weight.device))

    def free(self):
        self.H = None


# --------------------------------------------------------------------------- #
#  Driver: block-by-block sequential quantization
# --------------------------------------------------------------------------- #

def _get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, model.model.layers
    raise ValueError("Unsupported architecture: expected model.model.layers")


@torch.no_grad()
def gptq_quantize_model(model, calib_samples, bits: int, device,
                        sym: bool = True, groupsize: int = -1,
                        percdamp: float = 0.01, actorder: bool = False):
    """calib_samples: list of [1, seqlen] token tensors."""
    base, blocks = _get_blocks(model)
    dtype = next(model.parameters()).dtype
    # MPS has no float64 -> linalg on CPU; CUDA keeps it on-device (much faster).
    linalg_device = device if device.type == "cuda" else torch.device("cpu")

    # --- 1. capture inputs to block 0 ------------------------------------- #
    base.embed_tokens.to(device)
    if hasattr(base, "rotary_emb"):
        base.rotary_emb.to(device)

    inps, kwargs_store = [], {}

    class Catcher(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, hidden_states, **kwargs):
            inps.append(hidden_states.cpu())
            if not kwargs_store:
                for k, v in kwargs.items():
                    kwargs_store[k] = v.to(device) if torch.is_tensor(v) else v
            raise _Stop()

    class _Stop(Exception):
        pass

    blocks[0] = Catcher(blocks[0])
    for sample in calib_samples:
        try:
            model(sample.to(device))
        except _Stop:
            pass
    blocks[0] = blocks[0].block
    base.embed_tokens.cpu()
    torch.mps.empty_cache() if device.type == "mps" else None

    # --- 2. per block: fit Hessians, quantize, propagate --------------- #
    outs = [None] * len(inps)
    for b, block in enumerate(tqdm(blocks, desc="gptq blocks")):
        block.to(device)
        linears = {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
        gptq = {n: GPTQ(m, linalg_device=linalg_device) for n, m in linears.items()}

        handles = []
        for n, m in linears.items():
            def hook(mod, inp, out, name=n):
                gptq[name].add_batch(inp[0].detach())
            handles.append(m.register_forward_hook(hook))

        for j, inp in enumerate(inps):
            block(inp.to(device), **kwargs_store)
        for h in handles:
            h.remove()

        for n in linears:
            gptq[n].quantize(bits, sym=sym, groupsize=groupsize,
                             percdamp=percdamp, actorder=actorder)
            gptq[n].free()

        # re-run with quantized weights to feed the next block corrected activations
        for j, inp in enumerate(inps):
            out = block(inp.to(device), **kwargs_store)
            outs[j] = (out[0] if isinstance(out, tuple) else out).cpu()

        block.cpu()
        torch.mps.empty_cache() if device.type == "mps" else None
        inps, outs = outs, inps

    model.config.use_cache = False
    model.to(device)  # blocks were offloaded to CPU during quantization
    print(f"[gptq] done: {bits}-bit, groupsize={groupsize}, actorder={actorder}")
    return model
