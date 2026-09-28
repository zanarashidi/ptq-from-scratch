"""results/*.json -> results/ablation.png

Dumbbell chart: for each (method, grouping) config at a given bit-width, show the
no-rotation perplexity and the Hadamard-rotation perplexity connected by a bar,
with random-orthogonal and Givens-rotation results overlaid as secondary points
on the GPTQ per-channel row where available. The gap and its direction are the
result. Faceted into 4-bit and 3-bit panels.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

RESULTS = Path(__file__).resolve().parent.parent / "results"

INK = "#1a1a1a"
MUTED = "#9a9a9a"
NOROT = "#9aa0a6"      # no rotation - recessive
HAD = "#2f6f4f"        # hadamard - accent
RAND = "#b06a1f"       # random orthogonal - secondary point
GIVENS = "#5b4fa0"     # givens (pairwise) rotation - secondary point
GRID = "#e8e8e8"
BAD = "#b00020"

XMAX = 400            # threshold for "diverges" annotation vs. an actual point
TICK_CANDIDATES = [13, 15, 20, 25, 30, 40, 50, 75, 100, 130]

# (method, grouping) rows, top to bottom
ROWS = [
    ("rtn", -1, "RTN · per-channel"),
    ("gptq", -1, "GPTQ · per-channel"),
    ("gptq", 128, "GPTQ · group-128"),
]


def main():
    rows = [json.loads(p.read_text()) for p in RESULTS.glob("*.json")]
    fp16 = next((r["ppl"] for r in rows if r["method"] == "fp16"), None)

    def get(method, gs, bits, rot):
        return next((r["ppl"] for r in rows
                     if r["method"] == method and r.get("groupsize", -1) == gs
                     and r["bits"] == bits and r["rotation"] == rot), None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    npos = len(ROWS)
    seen = set()

    def lbl(name):
        if name in seen:
            return None
        seen.add(name)
        return name

    for ax, bits in zip(axes, (4, 3)):
        ypos = list(range(npos))[::-1]

        # gather this panel's own points first, so its axis range can be fit
        # tightly to just its own data - the two bit-widths don't need to share
        # a scale, and a wide shared range was crowding the close-together points
        points, panel_vals = [], [fp16] if fp16 else []
        for y, (method, gs, label) in zip(ypos, ROWS):
            p0, p1 = get(method, gs, bits, "none"), get(method, gs, bits, "hadamard")
            pr = get(method, gs, bits, "random")
            pg = get(method, gs, bits, "givens")
            divergent = bool(p0 and p1 and p0 > XMAX and p1 > XMAX)
            if not divergent:
                panel_vals += [v for v in (p0, p1, pr, pg) if v is not None]
            points.append((y, p0, p1, pr, pg, divergent))
        view = (min(panel_vals) / 1.15, max(panel_vals) * 1.2)

        for y, p0, p1, pr, pg, divergent in points:
            if divergent:
                ax.annotate(f"diverges  ({p0:,.0f} / {p1:,.0f} ppl)",
                            (view[1], y), xytext=(0, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=BAD)
                continue
            if p0 is None and p1 is None:
                continue

            if p0 and p1:
                ax.plot([p0, p1], [y, y], color=MUTED, lw=2, zorder=2,
                        solid_capstyle="round")
            if p0:
                ax.scatter([p0], [y], s=95, color=NOROT, edgecolor="white", lw=1.5,
                           zorder=3, label=lbl("no rotation"))
                ax.annotate(f"{p0:.1f}", (p0, y), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color=INK)
            if pr:
                ax.scatter([pr], [y], s=60, color=RAND, marker="D", edgecolor="white",
                           lw=1, zorder=4,
                           label=lbl("+ random orthogonal"))
            if pg:
                ax.scatter([pg], [y], s=70, color=GIVENS, marker="^", edgecolor="white",
                           lw=1, zorder=6,
                           label=lbl("+ Givens (pairwise)"))
                ax.annotate(f"{pg:.1f}", (pg, y), xytext=(0, 21),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color=GIVENS)
            if p1:
                ax.scatter([p1], [y], s=95, color=HAD, edgecolor="white", lw=1.5,
                           zorder=5, label=lbl("+ Hadamard"))
                ax.annotate(f"{p1:.1f}", (p1, y), xytext=(0, -17),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            color=HAD)

        if fp16:
            ax.axvline(fp16, color=INK, ls=(0, (4, 3)), lw=1.2, alpha=0.65, zorder=1)
            ax.annotate(f"fp16  {fp16:.1f}", (fp16, npos - 0.5), xytext=(5, 0),
                        textcoords="offset points", fontsize=8, color=INK, va="center")

        ticks = [t for t in TICK_CANDIDATES if view[0] <= t <= view[1]]

        ax.set_xscale("log")
        ax.set_xlim(*view)
        ax.set_ylim(-0.6, npos - 0.4)
        ax.set_yticks(list(range(npos))[::-1])
        ax.set_yticklabels([r[2] for r in ROWS], fontsize=9.5)
        ax.set_title(f"{bits}-bit weights", fontsize=11, color=INK, pad=8)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)

    axes[0].set_xlabel("WikiText-2 perplexity  (log scale)", fontsize=9)
    axes[1].set_xlabel("WikiText-2 perplexity  (log scale)", fontsize=9)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    order = ["no rotation", "+ Hadamard", "+ random orthogonal", "+ Givens (pairwise)"]
    pairs = sorted(zip(labels, handles), key=lambda t: order.index(t[0]))
    fig.legend([h for _, h in pairs], [l for l, _ in pairs], loc="lower center",
               ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Rotating weights before quantizing  ·  Qwen2.5-0.5B, WikiText-2",
                 y=1.04, fontsize=12.5, color=INK)
    fig.tight_layout(w_pad=3)
    out = RESULTS / "ablation.png"
    # dpi=220 rather than a lower value: GitHub's README view displays this at
    # roughly the CSS width of the content column, and on a HiDPI/retina screen
    # that needs real pixels well above the CSS size to render crisply - a lower
    # dpi looks fine opened directly but visibly softer embedded inline.
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
