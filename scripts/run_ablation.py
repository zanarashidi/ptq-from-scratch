"""Run the full grid and print the money table:

    method x bits x rotation  ->  WikiText-2 perplexity

Runs each config in a subprocess so MPS memory is released between runs.
"""
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# (method, bits, rotation, groupsize)
GRID = [
    ("fp16", 16, "none", -1),
    ("rtn", 8, "none", -1),
    ("rtn", 4, "none", -1),
    ("rtn", 4, "hadamard", -1),
    ("rtn", 3, "none", -1),
    ("rtn", 3, "hadamard", -1),
    ("gptq", 4, "none", -1),
    ("gptq", 4, "hadamard", -1),
    ("gptq", 3, "none", -1),
    ("gptq", 3, "hadamard", -1),
    ("gptq", 3, "random", -1),
    # Givens (composed pairwise rotations): no power-of-two constraint, unlike hadamard
    ("gptq", 4, "givens", -1),
    ("gptq", 3, "givens", -1),
    # group-wise scales (groupsize=128): makes 3-bit usable, tightens 4-bit
    ("gptq", 4, "none", 128),
    ("gptq", 4, "hadamard", 128),
    ("gptq", 3, "none", 128),
    ("gptq", 3, "hadamard", 128),
]


def main():
    extra = sys.argv[1:]  # forwarded, e.g. --model ... --limit-windows 20
    for method, bits, rot, gs in GRID:
        tag = f"{method}_b{bits}_{rot}_g{gs}"
        if (RESULTS / f"{tag}.json").exists():
            print(f"skip {tag} (exists)")
            continue
        cmd = [sys.executable, "-m", "scripts.run_experiment",
               "--method", method, "--bits", str(bits), "--rotation", rot,
               "--groupsize", str(gs)] + extra
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)

    rows = [json.loads(p.read_text()) for p in RESULTS.glob("*.json")]
    print(f"\n{'method':<6} {'bits':>4} {'group':>6} {'rotation':>9} {'ppl':>11} {'compress':>9}")
    print("-" * 52)
    # bits first, so configs at the same bit-width (the natural comparison) sit
    # together instead of being split across separate method blocks; rotation
    # ordered none -> hadamard -> random -> givens (baseline, then treatments)
    # rather than alphabetically
    rot_order = {"none": 0, "hadamard": 1, "random": 2, "givens": 3}
    for r in sorted(rows, key=lambda r: (-r["bits"], r.get("groupsize", -1),
                                         r["method"], rot_order.get(r["rotation"], 9))):
        c = r["bytes"].get("compression", 1.0)
        gs = r.get("groupsize", -1)
        print(f"{r['method']:<6} {r['bits']:>4} {gs:>6} {r['rotation']:>9} "
              f"{r['ppl']:>11.3f} {c:>8.2f}x")


if __name__ == "__main__":
    main()
