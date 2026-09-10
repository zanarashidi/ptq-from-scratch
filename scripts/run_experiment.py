"""Single quantization run -> perplexity. Building block for the ablation.

Examples
--------
  python -m scripts.run_experiment --method fp16
  python -m scripts.run_experiment --method rtn  --bits 4
  python -m scripts.run_experiment --method gptq --bits 3 --rotation hadamard
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ptq.model import DEFAULT_MODEL, get_device, load_model
from ptq.data import get_calibration
from ptq.rtn import quantize_model_rtn
from ptq.gptq import gptq_quantize_model
from ptq.rotation import apply_rotation
from ptq.eval_ppl import eval_ppl
from ptq.benchmark import theoretical_weight_bytes

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--method", choices=["fp16", "rtn", "gptq"], default="rtn")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--rotation", choices=["none", "hadamard", "random"], default="none")
    p.add_argument("--groupsize", type=int, default=-1)
    p.add_argument("--sym", action="store_true", default=True)
    p.add_argument("--asym", dest="sym", action="store_false")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--eval-seqlen", type=int, default=2048)
    p.add_argument("--limit-windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    device = get_device()
    print(f"device={device}  method={args.method}  bits={args.bits}  rotation={args.rotation}")

    model, tok = load_model(args.model)

    if args.rotation != "none":
        apply_rotation(model, mode=args.rotation, seed=args.seed)

    t0 = time.time()
    if args.method == "fp16":
        model.to(device)
    elif args.method == "rtn":
        model.to(device)
        quantize_model_rtn(model, args.bits, sym=args.sym, groupsize=args.groupsize)
    elif args.method == "gptq":
        model.to(device)
        calib = get_calibration(tok, args.nsamples, args.calib_seqlen, args.seed)
        gptq_quantize_model(model, calib, args.bits, device,
                            sym=args.sym, groupsize=args.groupsize)
    quant_time = time.time() - t0

    ppl = eval_ppl(model, tok, seqlen=args.eval_seqlen, device=device,
                   limit_windows=args.limit_windows)

    bytes_info = theoretical_weight_bytes(model, 16 if args.method == "fp16" else args.bits)

    rec = {
        "model": args.model, "method": args.method, "bits": args.bits,
        "rotation": args.rotation, "groupsize": args.groupsize, "sym": args.sym,
        "nsamples": args.nsamples, "calib_seqlen": args.calib_seqlen,
        "eval_seqlen": args.eval_seqlen, "ppl": ppl,
        "quant_seconds": round(quant_time, 1), "bytes": bytes_info,
    }
    print(json.dumps(rec, indent=2))

    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or f"{args.method}_b{args.bits}_{args.rotation}_g{args.groupsize}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote results/{tag}.json")


if __name__ == "__main__":
    main()
