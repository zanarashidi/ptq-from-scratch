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
        gs = r.get("groupsize", -1)
        gtag = "per-channel" if gs in (-1, 0) else f"g{gs}"
        key = f"{r['method']} {gtag} {r['rotation']}"
        series.setdefault(key, []).append((r["bits"], r["ppl"]))

    plt.figure(figsize=(8, 5.5))
    for key, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        style = "--o" if "hadamard" in key else (":s" if "random" in key else "-o")
        plt.plot(xs, ys, style, label=key, linewidth=2, markersize=7)

    if fp16:
        plt.axhline(fp16, color="k", linestyle="-", alpha=0.5, label=f"fp16 ({fp16:.1f})")

    # RTN 3-bit diverges to ~1e5+; clip the view to the region of interest and
    # flag the off-scale points instead of letting them flatten everything.
    ymax = 400
    offscale = [(key, b, p) for key, pts in series.items()
                for b, p in pts if p > ymax]
    for i, (key, b, p) in enumerate(sorted(offscale)):
        plt.annotate(f"{key.replace(' per-channel', '')} {b}b diverges ({p:,.0f})",
                     xy=(b, ymax), xytext=(-6, -14 - 12 * i),
                     textcoords="offset points", ha="left",
                     fontsize=7, color="crimson")
    plt.ylim(10, ymax)

    plt.xlabel("weight bits")
    plt.ylabel("WikiText-2 perplexity")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.gca().set_xticks([8, 4, 3])
    plt.legend(fontsize=8)
    plt.title("PTQ ablation: rotation vs no rotation (Qwen2.5-0.5B)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS / "ablation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
