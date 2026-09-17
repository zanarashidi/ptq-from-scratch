"""Fast unit tests - no model download. Run: python -m pytest -q"""
import torch

from ptq.quant import quantize_weight
from ptq.rotation import hadamard_matrix, random_orthogonal, givens_rotation, get_rotation


def test_rtn_error_shrinks_with_bits():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    errs = []
    for b in (8, 4, 3, 2):
        dq, _ = quantize_weight(w, b)
        errs.append((w - dq).pow(2).mean().item())
    assert all(errs[i] < errs[i + 1] for i in range(len(errs) - 1))


def test_rtn_exact_range():
    w = torch.randn(8, 32)
    dq, meta = quantize_weight(w, 4, sym=True)
    q = torch.round(w.float() / meta["scale"])
    assert q.abs().max() <= 7  # 4-bit symmetric -> [-8, 7]


def test_hadamard_orthogonal():
    for n in (2, 8, 64, 2048):
        H = hadamard_matrix(n)
        assert torch.allclose(H @ H.t(), torch.eye(n, dtype=H.dtype), atol=1e-9)


def test_hadamard_non_pow2_fallback_still_orthogonal():
    H = hadamard_matrix(896)  # Qwen2.5-0.5B hidden size
    assert torch.allclose(H @ H.t(), torch.eye(896, dtype=H.dtype), atol=1e-9)


def test_random_orthogonal():
    Q = random_orthogonal(128, seed=1)
    assert torch.allclose(Q @ Q.t(), torch.eye(128, dtype=Q.dtype), atol=1e-10)


def test_givens_orthogonal_including_odd_n():
    for n in (2, 5, 64, 896, 897):  # 897 exercises the odd-leftover path
        Q = givens_rotation(n, seed=0)
        assert torch.allclose(Q @ Q.t(), torch.eye(n, dtype=Q.dtype), atol=1e-9)


def test_givens_matches_random_orthogonal_on_non_pow2_n():
    """On Qwen's hidden=896 (not a power of two), hadamard_matrix() falls back
    to a partial kron() mix. givens_rotation() has no such constraint - with
    its default layer count it should reach full-rank mixing quality,
    matching a true random orthogonal matrix's outlier-smoothing (its actual
    target - not the partial Hadamard, which for any *single* outlier layout
    can land above or below either by chance)."""
    torch.manual_seed(0)
    n = 896
    w = torch.randn(256, n)
    outlier_idx = torch.randperm(n)[:24]
    w[:, outlier_idx] *= 15.0  # inject outlier channels, no periodic alignment
    ratio = lambda x: (x.abs().amax(1) / x.double().pow(2).mean(1).sqrt()).mean()

    r_none = ratio(w)
    r_random = ratio(w.double() @ random_orthogonal(n, seed=0))
    r_givens = ratio(w.double() @ givens_rotation(n, seed=0))

    assert r_givens < r_none  # still smooths outliers relative to no rotation
    assert abs(r_givens - r_random) < 0.1 * r_random  # within 10% of full random mixing


def test_rotation_reduces_outlier_dominance():
    """A weight matrix with a few fat channels: after rotation the per-row
    max/rms ratio (what sets the quant scale) should drop."""
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    w[:, ::37] *= 15.0  # inject outlier channels
    Q = get_rotation(256, "hadamard")
    wr = (w.double() @ Q)
    ratio = lambda x: (x.abs().amax(1) / x.double().pow(2).mean(1).sqrt()).mean()
    assert ratio(wr) < ratio(w)
