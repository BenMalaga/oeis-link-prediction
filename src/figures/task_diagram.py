"""Generate the task-design schematic for the README.

A small node-link diagram of real integer sequences (OEIS-style term lists as
node labels) connected by cross-reference edges. Solid edges are cross-references
that are KNOWN to the model at training time; the single dashed edge is a
HELD-OUT cross-reference the model must predict from the term lists alone.

This is a pure illustration of the task design. It contains no model outputs,
no scores, and no pre-registered outcome metrics. The term lists are the genuine
opening terms of the named sequences; the edges depict the kind of editorial
cross-reference the benchmark holds out and asks a term-only model to recover.

Run:  python -m src.figures.task_diagram
Out:  docs/figures/task_diagram.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Colorblind-friendly palette (Wong / Okabe-Ito) ----------------------------
INK = "#222222"
NODE_FILL = "#FFFFFF"
NODE_EDGE = "#0072B2"        # blue
KNOWN_EDGE = "#009E73"       # green  -> known cross-reference
HELDOUT_EDGE = "#D55E00"     # vermillion -> held-out cross-reference
MUTED = "#666666"

# --- Nodes: real OEIS sequences, genuine opening terms -------------------------
# label, sub-label (term list), (x, y)
NODES = {
    "A000045": ("Fibonacci\n0, 1, 1, 2, 3, 5, 8, 13, 21, 34", (0.20, 0.70)),
    "A000071": ("Fibonacci - 1\n0, 0, 1, 2, 4, 7, 12, 20, 33, 54", (0.20, 0.24)),
    "A000032": ("Lucas\n2, 1, 3, 4, 7, 11, 18, 29, 47, 76", (0.62, 0.78)),
    "A000204": ("Lucas (from 1)\n1, 3, 4, 7, 11, 18, 29, 47, 76", (0.90, 0.42)),
    "A000040": ("Primes\n2, 3, 5, 7, 11, 13, 17, 19, 23", (0.60, 0.14)),
}

# Solid edges = cross-references already present in the training graph.
KNOWN = [
    ("A000045", "A000032"),   # Fibonacci <-> Lucas (companion sequences)
    ("A000032", "A000204"),   # two offsets of the Lucas numbers
]

# Dashed edge = the held-out cross-reference the model must predict from terms.
HELDOUT = ("A000045", "A000071")  # A000071(n) = Fibonacci(n) - 1  (a transform-of link)


def _node_center(name):
    _, pos = NODES[name]
    return pos


def main():
    out = Path(__file__).resolve().parents[2] / "docs" / "figures" / "task_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 1.20)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # --- edges (drawn under the nodes) -----------------------------------------
    for a, b in KNOWN:
        (xa, ya), (xb, yb) = _node_center(a), _node_center(b)
        ax.add_patch(
            FancyArrowPatch(
                (xa, ya), (xb, yb),
                arrowstyle="-", color=KNOWN_EDGE, lw=2.4,
                shrinkA=46, shrinkB=46, zorder=1,
            )
        )

    a, b = HELDOUT
    (xa, ya), (xb, yb) = _node_center(a), _node_center(b)
    ax.add_patch(
        FancyArrowPatch(
            (xa, ya), (xb, yb),
            arrowstyle="-", color=HELDOUT_EDGE, lw=2.6,
            linestyle=(0, (4, 3)), shrinkA=46, shrinkB=46, zorder=1,
        )
    )
    # "?" marker on the held-out edge
    ax.text(
        (xa + xb) / 2 - 0.055, (ya + yb) / 2, "?",
        fontsize=20, color=HELDOUT_EDGE, fontweight="bold",
        ha="center", va="center", zorder=4,
    )

    # --- nodes ------------------------------------------------------------------
    for name, (sub, (x, y)) in NODES.items():
        w, h = 0.345, 0.155
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.8, edgecolor=NODE_EDGE, facecolor=NODE_FILL, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.034, name, fontsize=10.5, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4, family="monospace")
        ax.text(x, y - 0.026, sub, fontsize=8.0, color=MUTED,
                ha="center", va="center", zorder=4, family="monospace")

    # --- title / caption --------------------------------------------------------
    ax.text(
        0.0, 0.985,
        "Predicting OEIS cross-references from integer terms alone",
        fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top",
    )
    ax.text(
        0.0, 0.925,
        "Nodes are real sequences shown only by their opening terms. "
        "The model never sees names, formulas, or comments.",
        fontsize=9.5, color=MUTED, ha="left", va="top",
    )

    # --- legend -----------------------------------------------------------------
    handles = [
        Line2D([0], [0], color=KNOWN_EDGE, lw=2.6,
               label="Known cross-reference (in training graph)"),
        Line2D([0], [0], color=HELDOUT_EDGE, lw=2.6, linestyle=(0, (4, 3)),
               label="Held-out edge the model must predict"),
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
        frameon=False, fontsize=9.0, handlelength=2.4,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
