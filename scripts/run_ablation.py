"""Run the full grid and print the money table:

    method x bits x rotation  ->  WikiText-2 perplexity

Runs each config in a subprocess so MPS memory is released between runs.
"""
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

GRID = [
    ("fp16", 16, "none"),
    ("rtn", 8, "none"),
    ("rtn", 4, "none"),
    ("rtn", 4, "hadamard"),
    ("rtn", 3, "none"),
    ("rtn", 3, "hadamard"),
    ("gptq", 4, "none"),
    ("gptq", 4, "hadamard"),
    ("gptq", 3, "none"),
    ("gptq", 3, "hadamard"),
    ("gptq", 3, "random"),
]


def main():
    extra = sys.argv[1:]  # forwarded, e.g. --model ... --limit-windows 20
    for method, bits, rot in GRID:
        tag = f"{method}_b{bits}_{rot}_g-1"
        if (RESULTS / f"{tag}.json").exists():
            print(f"skip {tag} (exists)")
            continue
        cmd = [sys.executable, "-m", "scripts.run_experiment",
               "--method", method, "--bits", str(bits), "--rotation", rot] + extra
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)

    rows = sorted(json.loads(p.read_text()) for p in RESULTS.glob("*.json"))
    print(f"\n{'method':<6} {'bits':>4} {'rotation':>9} {'ppl':>9} {'compress':>9}")
    print("-" * 42)
    for r in sorted(rows, key=lambda r: (r["method"], -r["bits"], r["rotation"])):
        c = r["bytes"].get("compression", 1.0)
        print(f"{r['method']:<6} {r['bits']:>4} {r['rotation']:>9} "
              f"{r['ppl']:>9.3f} {c:>8.2f}x")


if __name__ == "__main__":
    main()
