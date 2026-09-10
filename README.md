# PTQ from scratch: a rotation ablation

Post-training quantization of a small LLM, implemented from first principles, to
reproduce the central finding of **QuaRot** (Ashkboos et al., 2024): rotating the
weights and activations of a transformer by an orthogonal matrix before
quantizing removes the outlier channels that otherwise force a coarse
quantization grid, and this is what makes aggressive low-bit (3-4 bit)
quantization work.

Nothing here wraps `bitsandbytes` or `AutoGPTQ`. The RTN quantizer, the GPTQ
inverse-Hessian update, and the rotation fusion are all written out so every line
is explainable.

## What's implemented

| Piece | File | Notes |
|---|---|---|
| RTN baseline | `ptq/rtn.py`, `ptq/quant.py` | per-channel / per-group, symmetric or asymmetric. The control group. |
| GPTQ | `ptq/gptq.py` | Hessian from calibration activations, Cholesky-based inverse, column-wise OBQ error compensation (Algorithm 1 + 2 of the paper). Block-by-block so only one block's Hessians are resident. |
| Rotation (mini-QuaRot) | `ptq/rotation.py` | R1: fuse RMSNorm scales into adjacent linears, then rotate the residual stream by a Hadamard (or random orthogonal) matrix. Function-preserving via computational invariance. |
| Perplexity | `ptq/eval_ppl.py` | WikiText-2, non-overlapping windows - the standard PTQ metric. |
| Memory / throughput | `ptq/benchmark.py` | theoretical bytes-streamed-per-token + measured tokens/s. |

R2/R3/R4 from QuaRot (head-wise value rotation, online Hadamard before
`down_proj`) and KV-cache quantization are **not** done - R1 alone already shows
the effect. See "Extensions".

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # MPS lacks a few linalg ops
```

Default model is `Qwen/Qwen2.5-0.5B` - small enough to run the whole pipeline on
an M2/8GB laptop in minutes. The GPTQ linear algebra runs on CPU in float64
regardless of device.

## Run

```bash
# one config
python -m scripts.run_experiment --method fp16
python -m scripts.run_experiment --method rtn  --bits 4
python -m scripts.run_experiment --method gptq --bits 3 --rotation hadamard

# the whole ablation grid -> results/*.json + a summary table
python -m scripts.run_ablation
# fast pass while iterating:
python -m scripts.run_ablation --limit-windows 20 --nsamples 32
```

Each run writes `results/<tag>.json`. `run_ablation.py` runs every config in a
subprocess (so MPS memory is released) and prints:

```
method  bits  rotation       ppl  compress
------------------------------------------
fp16      16       none    <fill>     1.00x
rtn        4       none    <fill>     4.00x
rtn        4   hadamard    <fill>     4.00x
gptq       3       none    <fill>     5.33x
gptq       3   hadamard    <fill>     5.33x
...
```

## The result (fill in after running)

_Ablation table / plot here. The row that matters: **GPTQ 3-bit, no rotation**
vs **GPTQ 3-bit + Hadamard** - the rotation should recover most of the gap back
to fp16, while at 4-bit the effect is smaller because the grid is already fine
enough to tolerate the outliers._

### Pipeline smoke check (Qwen2.5-0.5B, 3 eval windows, 16 calib samples - NOT final numbers)

| method | bits | WikiText-2 ppl |
|---|---|---|
| fp16 | 16 | 13.85 |
| RTN | 8 | 13.87 |
| RTN | 4 | 33.81 |
| GPTQ | 4 | 18.63 |

GPTQ already cuts the 4-bit gap by more than half with almost no calibration
data. Run the full grid (`scripts/run_ablation.py`, 128 samples, full eval) for
the real table.

## Why rotation works (the interview answer)

_3-4 paragraphs, written after seeing the numbers:_

1. **The outlier problem.** A handful of activation channels in a trained
   transformer carry magnitudes 10-100x the rest. Quantization scale is set by
   the max value in a row/group, so one outlier stretches the grid and every
   ordinary value gets 1-2 effective bits.

2. **What a rotation does.** An orthogonal `Q` mixes every channel into every
   other. A concentrated spike becomes spread across all coordinates - by a
   Johnson-Lindenstrauss / concentration argument the post-rotation values look
   near-Gaussian with a much smaller max/RMS ratio. A Hadamard matrix is the
   cheap structured choice: `O(n log n)`, entries `±1/√n`.

3. **Why it's free at inference (computational invariance).** RMSNorm is
   invariant to orthogonal rotation of its input (it only divides by the norm).
   So if you rotate everything entering the residual stream by `Q` and everything
   leaving it by `Qᵀ`, and fold the norm scales into the weights first, the
   network computes the identical function - the rotation is a change of basis
   absorbed entirely into the weight matrices offline.

4. **Why this matters for memory-bound inference.** LLM decode is bandwidth-bound:
   time per token ≈ (bytes of weights + KV cache) / HBM bandwidth. Going 16→4 bit
   is a ~4x cut in bytes streamed per token. Rotation is what lets you take that
   cut without the accuracy falling off a cliff.

## Extensions

- KV-cache int8/int4 (the long-context bottleneck; gestures at TurboQuant)
- R2/R3/R4 rotations for activation-quantization (W4A4)
- packed int4 storage + a real fast kernel (Marlin) for a true tokens/s number
- second data point on Llama-3.2-1B (hidden=2048, a clean power-of-two Hadamard)

## References

- Frantar et al., *GPTQ*, 2023
- Ashkboos et al., *QuaRot*, 2024
- Tseng et al., *QuIP#* / Chee et al., *QuIP*, 2023-24 (incoherence processing)
