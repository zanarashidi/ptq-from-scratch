"""results/*.json -> results/ablation_4bit.png, results/ablation_3bit.png (+ _dark variants)

Two standalone dumbbell charts, one per bit-width, each showing the
no-rotation perplexity and the Hadamard-rotation perplexity connected by a
bar, with random-orthogonal and Givens-rotation results overlaid as secondary
points on the GPTQ per-channel row where available. Split rather than faceted
in one figure so each gets its own tight axis range and stands alone in the
README. Each is rendered twice (light/dark) so a `<picture>` element can match
the viewer's GitHub theme instead of a light chart looking stranded on a dark
page.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

RESULTS = Path(__file__).resolve().parent.parent / "results"

# categorical colors: reference-palette slots 1-3 (blue/orange/aqua), the
# subset validated all-pairs CVD-safe in both light and dark (palette.md) -
# these three points get compared against each other, not just neighbors.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", baseline="#c3c2b7",
        norot="#898781", had="#2a78d6", rand="#eb6834", givens="#1baf7a",
        bad="#d03b3b",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", baseline="#383835",
        norot="#898781", had="#3987e5", rand="#d95926", givens="#199e70",
        bad="#e66767",
    ),
}

XMAX = 400  # threshold for "diverges" annotation vs. an actual point
TICK_CANDIDATES = [13, 15, 20, 25, 30, 40, 50, 75, 100, 130]

# (method, grouping) rows, top to bottom
ROWS = [
    ("rtn", -1, "RTN · per-channel"),
    ("gptq", -1, "GPTQ · per-channel"),
    ("gptq", 128, "GPTQ · group-128"),
]


def render(bits, theme_name, rows, fp16):
    c = THEMES[theme_name]
    npos = len(ROWS)

    def get(method, gs, rot):
        return next((r["ppl"] for r in rows
                     if r["method"] == method and r.get("groupsize", -1) == gs
                     and r["bits"] == bits and r["rotation"] == rot), None)

    points, panel_vals = [], [fp16] if fp16 else []
    ypos = list(range(npos))[::-1]
    for y, (method, gs, label) in zip(ypos, ROWS):
        p0, p1 = get(method, gs, "none"), get(method, gs, "hadamard")
        pr, pg = get(method, gs, "random"), get(method, gs, "givens")
        divergent = bool(p0 and p1 and p0 > XMAX and p1 > XMAX)
        if not divergent:
            panel_vals += [v for v in (p0, p1, pr, pg) if v is not None]
        points.append((y, p0, p1, pr, pg, divergent))
    view = (min(panel_vals) / 1.15, max(panel_vals) * 1.2)

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    seen = set()

    def lbl(name):
        if name in seen:
            return None
        seen.add(name)
        return name

    for y, p0, p1, pr, pg, divergent in points:
        if divergent:
            ax.annotate(f"diverges  ({p0:,.0f} / {p1:,.0f} ppl)",
                        (view[1], y), xytext=(0, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=c["bad"])
            continue
        if p0 is None and p1 is None:
            continue

        if p0 and p1:
            ax.plot([p0, p1], [y, y], color=c["muted"], lw=2, zorder=2,
                     solid_capstyle="round")
        if p0:
            ax.scatter([p0], [y], s=95, color=c["norot"], edgecolor=c["surface"],
                       lw=1.5, zorder=3, label=lbl("no rotation"))
            # value labels use text ink, never a series hue - the colored dot
            # beside them (via vertical position, above vs below) carries identity
            ax.annotate(f"{p0:.1f}", (p0, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=c["ink"])
        if pr:
            ax.scatter([pr], [y], s=60, color=c["rand"], marker="D",
                       edgecolor=c["surface"], lw=1, zorder=4,
                       label=lbl("+ random orthogonal"))
        if pg:
            ax.scatter([pg], [y], s=70, color=c["givens"], marker="^",
                       edgecolor=c["surface"], lw=1, zorder=6,
                       label=lbl("+ Givens (pairwise)"))
            ax.annotate(f"{pg:.1f}", (pg, y), xytext=(0, 21),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=c["secondary"])
        if p1:
            ax.scatter([p1], [y], s=95, color=c["had"], edgecolor=c["surface"],
                       lw=1.5, zorder=5, label=lbl("+ Hadamard"))
            ax.annotate(f"{p1:.1f}", (p1, y), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=c["secondary"])

    if fp16:
        ax.axvline(fp16, color=c["ink"], ls=(0, (4, 3)), lw=1.2, alpha=0.65, zorder=1)
        ax.annotate(f"fp16  {fp16:.1f}", (fp16, npos - 0.5), xytext=(5, 0),
                    textcoords="offset points", fontsize=8, color=c["ink"], va="center")

    ticks = [t for t in TICK_CANDIDATES if view[0] <= t <= view[1]]

    ax.set_xscale("log")
    ax.set_xlim(*view)
    ax.set_ylim(-0.6, npos - 0.4)
    ax.set_yticks(list(range(npos))[::-1])
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=9.5, color=c["ink"])
    ax.set_title(f"{bits}-bit weights", fontsize=12, color=c["ink"], pad=10)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelsize=8.5, colors=c["muted"])
    ax.grid(axis="x", color=c["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["baseline"])
    ax.tick_params(left=False)
    ax.set_xlabel("WikiText-2 perplexity  (log scale)", fontsize=9, color=c["secondary"])

    fig.suptitle("Rotating weights before quantizing · Qwen2.5-0.5B",
                 y=0.98, fontsize=11.5, color=c["ink"])
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))

    handles, labels = ax.get_legend_handles_labels()
    order = ["no rotation", "+ Hadamard", "+ random orthogonal", "+ Givens (pairwise)"]
    pairs = sorted(zip(labels, handles), key=lambda t: order.index(t[0]))
    leg = fig.legend([h for _, h in pairs], [l for l, _ in pairs], loc="lower center",
                     ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    for text in leg.get_texts():
        text.set_color(c["secondary"])
    suffix = "" if theme_name == "light" else "_dark"
    out = RESULTS / f"ablation_{bits}bit{suffix}.png"
    # dpi=220: fine opened directly at a lower dpi, but visibly soft embedded
    # in the README, where GitHub displays it at a HiDPI-scaled CSS width.
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = [json.loads(p.read_text()) for p in RESULTS.glob("*.json")]
    fp16 = next((r["ppl"] for r in rows if r["method"] == "fp16"), None)
    for bits in (4, 3):
        for theme_name in ("light", "dark"):
            render(bits, theme_name, rows, fp16)


if __name__ == "__main__":
    main()
