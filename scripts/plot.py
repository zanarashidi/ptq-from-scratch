"""results/*.json -> results/ablation.png : perplexity vs bit-width, per method."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    rows = [json.loads(p.read_text()) for p in RESULTS.glob("*.json")]
    fp16 = next((r["ppl"] for r in rows if r["method"] == "fp16"), None)

    series = {}
    for r in rows:
        if r["method"] == "fp16":
            continue
        key = f"{r['method']} {r['rotation']}"
        series.setdefault(key, []).append((r["bits"], r["ppl"]))

    plt.figure(figsize=(7, 5))
    for key, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        style = "--o" if "hadamard" in key else (":s" if "random" in key else "-o")
        plt.plot(xs, ys, style, label=key, linewidth=2, markersize=7)

    if fp16:
        plt.axhline(fp16, color="k", linestyle="-", alpha=0.5, label="fp16")

    plt.xlabel("weight bits")
    plt.ylabel("WikiText-2 perplexity")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title("PTQ ablation: rotation vs no rotation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS / "ablation.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
