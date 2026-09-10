#!/usr/bin/env bash
# One-shot setup + full ablation on a fresh RunPod CUDA pod.
#
#   git clone https://github.com/zanarashidi/ptq-from-scratch.git
#   cd ptq-from-scratch && bash scripts/runpod_setup.sh
#
# The PyTorch pod images already ship a CUDA-enabled torch, so we don't reinstall it.
set -euo pipefail

python -m pip install -q --upgrade pip
python -m pip install -q "transformers>=4.45" "datasets>=2.20" numpy tqdm

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# No MPS on the pod -> get_device() falls through to CUDA automatically.
# GPU has float64, so the .cpu().double() linalg paths still work (just run on CPU; fine for 0.5B).
# Paper-matching calibration now that memory isn't the constraint:
python -m scripts.run_ablation --nsamples 128 --calib-seqlen 2048

python -m scripts.plot
echo "Done. Pull results/ back with:  scp -r <pod>:$(pwd)/results ./results"
