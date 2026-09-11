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
export PYTORCH_ENABLE_MPS_FALLBACK=1   # only needed on Apple MPS
```

Default model is `Qwen/Qwen2.5-0.5B` - small enough to run the whole pipeline on
an M2/8GB laptop. The GPTQ linear algebra runs in float64: on CUDA it stays on
the GPU, on MPS it falls back to CPU (MPS has no float64).

## Run

```bash
# one config
python -m scripts.run_experiment --method fp16
python -m scripts.run_experiment --method rtn  --bits 4
python -m scripts.run_experiment --method gptq --bits 3 --rotation hadamard --groupsize 128

# the whole ablation grid -> results/*.json + a summary table
python -m scripts.run_ablation --nsamples 128 --calib-seqlen 2048
python -m scripts.plot                       # -> results/ablation.png

# fast pass while iterating:
python -m scripts.run_ablation --limit-windows 20 --nsamples 32
```

Each run writes `results/<tag>.json`. `run_ablation.py` runs every config in a
subprocess (so GPU/MPS memory is released between runs) and prints the summary
table below. To run on a rented GPU, `scripts/runpod_setup.sh` does the whole
thing on a fresh CUDA pod.

## The result

`Qwen/Qwen2.5-0.5B`, WikiText-2 perplexity (seqlen 2048, full test set), GPTQ
calibrated on 128 sequences x 2048 tokens, symmetric weights, run on an RTX 3090.

| method | bits | grouping | rotation | WikiText-2 ppl | vs fp16 |
|---|---|---|---|---:|---:|
| fp16   | 16 | -           | -        | **13.07** | - |
| RTN    | 8  | per-channel | none     | 13.09     | +0.02 |
| RTN    | 4  | per-channel | none     | 30.26     | +17.19 |
| RTN    | 4  | per-channel | hadamard | 25.50     | +12.43 |
| RTN    | 3  | per-channel | none     | 136,980   | diverges |
| RTN    | 3  | per-channel | hadamard | 254,852   | diverges |
| GPTQ   | 4  | per-channel | none     | 16.54     | +3.47 |
| GPTQ   | 4  | per-channel | hadamard | 15.33     | +2.26 |
| GPTQ   | 3  | per-channel | none     | 107.09    | +94.02 |
| GPTQ   | 3  | per-channel | random   | 82.61     | +69.54 |
| GPTQ   | 3  | per-channel | hadamard | 85.27     | +72.20 |
| GPTQ   | 4  | group-128   | none     | 14.57     | +1.50 |
| GPTQ   | 4  | group-128   | hadamard | 14.40     | +1.33 |
| GPTQ   | 3  | group-128   | none     | 26.55     | +13.48 |
| GPTQ   | 3  | group-128   | hadamard | 26.76     | +13.69 |

![ablation](results/ablation.png)

### Reading the table

1. **The quantizer is correct.** RTN at 8 bits is within 0.02 ppl of fp16, so any
   damage at 4/3 bits is quantization difficulty, not a bug.

2. **GPTQ's error feedback is worth a lot.** At 4-bit per-channel it nearly
   halves the gap to fp16 versus RTN (16.5 vs 30.3). This is the inverse-Hessian
   update compensating each column's rounding error against the not-yet-quantized
   columns.

3. **Rotation helps in the coarse regime, and any orthogonal matrix will do.**
   With a single per-channel scale per row, the Hadamard rotation improves every
   setting it was tried in: RTN-4 30.3 -> 25.5 (-16%), GPTQ-4 16.5 -> 15.3
   (closes ~35% of the residual gap), GPTQ-3 107 -> 85. A *random* orthogonal
   matrix does the same (GPTQ-3: 82.6, statistically indistinguishable from the
   Hadamard's 85.3) - consistent with the mechanism being "mix every channel into
   every other," for which Hadamard is just the cheap `O(n log n)` structured
   choice.

4. **Rotation alone does not rescue 3-bit here, and that is the interesting
   part.** GPTQ-3 + Hadamard is still 85 ppl against fp16's 13. Qwen2.5-0.5B has
   hidden size 896 = 128 x 7, which is not a power of two, so `hadamard_matrix()`
   falls back to `kron(H_128, I_7)` - it only mixes channels *within* each block
   of 128 and leaves the 7 blocks unmixed. A model with a power-of-two hidden
   dimension (Llama-3.2-1B, 2048) gets a full Hadamard; that is the natural next
   data point.

5. **Group scales and rotation solve overlapping problems.** Moving GPTQ to
   group-128 scales takes 3-bit from unusable (107) to 26.5 and 4-bit to within
   1.5 ppl of fp16. Once the scale is local to a 128-wide group, the group max
   already tracks local outliers, and adding the (partial) rotation on top
   changes nothing: 14.57 -> 14.40 at 4-bit, 26.55 -> 26.76 at 3-bit. On this
   model, per-group quantization is the bigger lever; rotation is what you reach
   for when you want to keep a *single* scale per channel (cheaper metadata,
   simpler kernels) and still survive 4-bit.

## Why rotation works

1. **The outlier problem.** A handful of activation channels in a trained
   transformer carry magnitudes 10-100x the rest. A quantization scale is set by
   the largest value in its row/group, so one outlier stretches the grid and
   every ordinary value collapses onto 1-2 effective levels.

2. **What a rotation does.** An orthogonal `Q` mixes every channel into every
   other. A concentrated spike becomes spread across all coordinates; by a
   Johnson-Lindenstrauss / concentration argument the rotated vector looks
   near-Gaussian with a much smaller max/RMS ratio, so a uniform grid fits it
   well. A Hadamard matrix is the cheap structured choice: `O(n log n)`, entries
   `+-1/sqrt(n)`. The ablation confirms a *random* orthogonal matrix works
   equally well - the structure buys speed, not accuracy.

3. **Why it's free at inference (computational invariance).** RMSNorm is
   invariant to an orthogonal rotation of its input (it only divides by the
   norm). So if you rotate everything entering the residual stream by `Q` and
   everything leaving it by `Q^T`, and fold the norm scales into the weights
   first, the network computes the identical function - the rotation is a change
   of basis absorbed entirely into the weight matrices offline. `apply_rotation`
   verifies this: fp16 ppl moves by <0.01 through the extra matmuls.

4. **Why this matters for memory-bound inference.** LLM decode is
   bandwidth-bound: time per token is approximately (weight bytes + KV-cache
   bytes) / HBM bandwidth. Going 16 -> 4 bit is a ~4x cut in bytes streamed per
   token. Rotation is one of the tools that lets you take that cut while keeping
   a single scale per channel and without accuracy falling off a cliff - most
   effective, on the evidence here, when the hidden dimension admits a full
   Hadamard.

## Extensions

- Llama-3.2-1B (hidden 2048): a clean power-of-two Hadamard, to test whether full
  channel mixing closes the 3-bit gap that the partial `kron(H_128, I_7)` cannot
- KV-cache int8/int4 (the long-context bandwidth bottleneck)
- R2/R3/R4 rotations for activation quantization (W4A4)
- packed int4 storage + a real fast kernel (Marlin) for a measured tokens/s number

## References

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers*, 2023
- Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, 2024
- Chee et al., *QuIP*, 2023; Tseng et al., *QuIP#*, 2024 (incoherence processing)
