"""Rotation preprocessing - the mini-QuaRot (Ashkboos et al. 2024).

Core claim being reproduced: multiplying the residual stream by an orthogonal
matrix Q before quantizing spreads the few large-magnitude "outlier" channels
across all channels, so no single weight/activation value dominates its row's
quantization scale. Because Q is orthogonal and RMSNorm is scale-invariant, the
network computes the *same function* (computational invariance) - the rotation is
fused offline into the surrounding weights and costs nothing at inference.

This module implements R1 only: a single global rotation of the residual stream.
QuaRot's R2/R3/R4 (head-wise value rotation, online Hadamard before down_proj)
are left as TODOs - R1 alone already demonstrates the outlier effect.
"""
import math
import warnings

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Orthogonal matrices
# --------------------------------------------------------------------------- #

def _sylvester(n: int) -> torch.Tensor:
    assert n & (n - 1) == 0, "sylvester needs a power of two"
    H = torch.ones((1, 1), dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard. For non-power-of-two n, fall back to
    kron(H_{2^k}, I_m) with 2^k the largest power of two dividing n."""
    if n & (n - 1) == 0:
        return _sylvester(n)
    k = n & (-n)  # largest power of two dividing n
    m = n // k
    warnings.warn(
        f"hidden size {n} is not a power of two; using kron(H_{k}, I_{m}). "
        f"Outlier smoothing will be partial. Llama-3.2-1B (2048) avoids this."
    )
    return torch.kron(_sylvester(k), torch.eye(m, dtype=torch.float64))


def random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((n, n), generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q *= torch.sign(torch.diag(r))  # make it deterministic / proper
    return q


def givens_rotation(n: int, layers: int = None, seed: int = 0) -> torch.Tensor:
    """Compose random pairwise (Givens) plane rotations into an n x n orthogonal
    matrix - ParoQuant's (2025) alternative to a global Hadamard.

    hadamard_matrix() needs n a power of two and falls back to a *partial*
    kron() mix when it isn't (as on Qwen2.5-0.5B, hidden=896). A Givens
    rotation has no such constraint: each layer randomly pairs up all n
    coordinates and rotates every pair by an independent angle. Composing
    enough layers (product of orthogonal matrices is orthogonal) mixes every
    coordinate with many others, same effect as a full Hadamard, for any n.
    """
    g = torch.Generator().manual_seed(seed)
    Q = torch.eye(n, dtype=torch.float64)
    # A *random* pairing each round mixes more slowly than a structured
    # butterfly (FFT/Hadamard's fixed log2(n)-stage wiring): empirically this
    # needs ~3x as many rounds to converge to full-rank mixing (verified by
    # sweeping outlier-ratio vs layer count until it matches random_orthogonal).
    layers = layers or max(32, 3 * int(math.ceil(math.log2(n))))

    for _ in range(layers):
        perm = torch.randperm(n, generator=g)
        angles = torch.rand(n // 2, generator=g, dtype=torch.float64) * (2 * math.pi)

        a = perm[0::2][: n // 2]  # first index of each pair
        b = perm[1::2][: n // 2]  # second index of each pair
        cos, sin = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
        Qa, Qb = Q[a], Q[b]
        Q[a] = cos * Qa - sin * Qb
        Q[b] = sin * Qa + cos * Qb
        # if n is odd, perm[-1] has no partner and sits out this layer.

    return Q


def get_rotation(n: int, mode: str, seed: int = 0) -> torch.Tensor:
    if mode == "hadamard":
        return hadamard_matrix(n)
    if mode == "random":
        return random_orthogonal(n, seed)
    if mode == "givens":
        return givens_rotation(n, seed=seed)
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
#  Fusion into a Llama/Qwen2-style model
# --------------------------------------------------------------------------- #

def _rms_names(layer):
    return layer.input_layernorm, layer.post_attention_layernorm


@torch.no_grad()
def _fold_rmsnorm_into(linear: nn.Linear, norm_weight: torch.Tensor):
    """W <- W @ diag(gamma); caller then sets gamma := 1."""
    linear.weight.data.mul_(norm_weight.to(linear.weight.dtype).unsqueeze(0))


@torch.no_grad()
def untie_lm_head(model):
    """Qwen2.5-0.5B ties lm_head.weight to embed_tokens.weight. Rotation and
    norm-folding must touch them independently, so give lm_head its own copy."""
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.data.clone(), requires_grad=False
        )
        model.config.tie_word_embeddings = False


@torch.no_grad()
def fuse_layernorms(model):
    """Absorb every RMSNorm scale into the following linear(s) so the norms
    become plain (weight-1) RMSNorm - a prerequisite for rotation invariance."""
    untie_lm_head(model)
    m = model.model
    for layer in m.layers:
        pre, post = _rms_names(layer)
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            _fold_rmsnorm_into(lin, pre.weight.data)
        pre.weight.data.fill_(1.0)
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj):
            _fold_rmsnorm_into(lin, post.weight.data)
        post.weight.data.fill_(1.0)
    _fold_rmsnorm_into(model.lm_head, m.norm.weight.data)
    m.norm.weight.data.fill_(1.0)


@torch.no_grad()
def apply_rotation(model, mode: str = "hadamard", seed: int = 0):
    """Fuse layernorms, then rotate the residual stream by Q. Function-preserving."""
    if mode == "none":
        return model
    fuse_layernorms(model)
    m = model.model
    hidden = model.config.hidden_size
    Q = get_rotation(hidden, mode, seed)

    Qc = Q.cpu()  # MPS has no float64; do the fusion math on CPU

    def rmul(linear):  # reads residual:  W <- W @ Q
        w = linear.weight
        w.data = (w.data.cpu().double() @ Qc).to(w.dtype).to(w.device)

    def lmul(linear):  # writes residual: W <- Q^T @ W
        w = linear.weight
        w.data = (Qc.t() @ w.data.cpu().double()).to(w.dtype).to(w.device)

    rmul(m.embed_tokens)  # embedding writes into the residual stream

    for layer in m.layers:
        rmul(layer.self_attn.q_proj)
        rmul(layer.self_attn.k_proj)
        rmul(layer.self_attn.v_proj)
        lmul(layer.self_attn.o_proj)
        rmul(layer.mlp.gate_proj)
        rmul(layer.mlp.up_proj)
        lmul(layer.mlp.down_proj)

    rmul(model.lm_head)
    print(f"[rotation] applied R1 ({mode}) to residual stream, hidden={hidden}")
    return model
