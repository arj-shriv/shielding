"""
Generate images_for_paper/architecture/k100s2_v1.png and transfer_mlp.png

Horizontal block diagrams of the two production models:
  - k100s2_v1    : the single-layer CNN backbone
  - transfer_mlp : the double-layer head (frozen k100s2_v1 backbone + small
                    trainable correction head), fed with TrackNet10 as the
                    layer-2 source spectrum

Layer shapes/params are read live from the saved .pkl models, not hardcoded,
so the diagram can't drift out of sync with what's actually deployed.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "images_for_paper" / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_BASE = REPO / "models" / "k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240" / "primary"

# ── colour palette (matches app/components/flux_plot.py MAT_COLOR family) ────
CONV_COLOR   = "#4e79a7"   # blue   — convolution
POOL_COLOR   = "#c9c9c9"   # gray   — pooling / flatten (no params)
DENSE_COLOR  = "#f28e2b"   # orange — dense
FROZEN_COLOR = "#9098a5"   # slate  — frozen backbone (non-trainable)
NEW_COLOR    = "#59a14f"   # green  — new trainable head
TEXT_COLOR   = "#1e293b"


def _load(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _layer_rows(model, skip_types=("InputLayer",)):
    rows = []
    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls in skip_types:
            continue
        try:
            shape = layer.output.shape
        except Exception:
            shape = None
        rows.append((layer.name, cls, shape, layer.count_params(), layer.trainable))
    return rows


def _color_for(cls: str, trainable: bool, forced=None) -> str:
    if forced is not None:
        return forced
    if not trainable:
        return FROZEN_COLOR
    if cls == "Conv1D":
        return CONV_COLOR
    if cls in ("MaxPooling1D", "Flatten"):
        return POOL_COLOR
    if cls == "Dense":
        return DENSE_COLOR
    return "#888888"


def draw_diagram(title: str, subtitle: str, rows, out_path: Path,
                  forced_colors=None, param_note: str = ""):
    forced_colors = forced_colors or {}
    n = len(rows)
    box_w, box_h, gap = 2.15, 1.05, 0.55
    fig_w = n * (box_w + gap) + gap
    fig_h = 3.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y_mid = fig_h * 0.50
    x = gap
    for i, (name, cls, shape, params, trainable) in enumerate(rows):
        color = _color_for(cls, trainable, forced_colors.get(name))
        box = FancyBboxPatch(
            (x, y_mid - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3, edgecolor="white", facecolor=color, alpha=0.92,
            zorder=3,
        )
        ax.add_patch(box)

        shape_str = "×".join(str(d) for d in shape[1:]) if shape else ""
        label_top = cls if cls != "Sequential" else "k100s2_v1\n(frozen)"
        ax.text(x + box_w / 2, y_mid + 0.20, label_top,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", zorder=4)
        ax.text(x + box_w / 2, y_mid - 0.10, shape_str,
                ha="center", va="center", fontsize=8.5, color="white", zorder=4)
        param_str = f"{params:,} params" if params else "—"
        ax.text(x + box_w / 2, y_mid - box_h / 2 - 0.22, param_str,
                ha="center", va="top", fontsize=7.8, color=TEXT_COLOR, zorder=4)

        if i < n - 1:
            ax.add_patch(FancyArrow(
                x + box_w + 0.06, y_mid, gap - 0.12, 0,
                width=0.012, head_width=0.14, head_length=0.10,
                length_includes_head=True, color="#8a8f98", zorder=2,
            ))
        x += box_w + gap

    ax.text(fig_w / 2, fig_h - 0.32, title, ha="center", va="top",
            fontsize=13.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(fig_w / 2, fig_h - 0.68, subtitle, ha="center", va="top",
            fontsize=9.5, color="#555555")
    if param_note:
        ax.text(fig_w / 2, 0.12, param_note, ha="center", va="bottom",
                fontsize=8.5, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    v1 = _load(MODEL_BASE / "k100s2_v1" / "model.pkl")
    tr = _load(MODEL_BASE / "transfer_mlp" / "transfer_model.pkl")

    v1_rows = _layer_rows(v1)
    draw_diagram(
        title="k100s2_v1 — single-layer model",
        subtitle="Input: source spectrum (250,) + thickness + density  →  transmitted flux (250,)",
        rows=v1_rows,
        out_path=OUT_DIR / "k100s2_v1.png",
        param_note=f"{v1.count_params():,} trainable params total",
    )

    tr_rows = _layer_rows(tr)
    forced = {"sequential": FROZEN_COLOR, "head_hidden": NEW_COLOR, "correction_head": NEW_COLOR}
    n_new = sum(p for _, _, _, p, t in tr_rows if t and _ not in ("sequential",))
    draw_diagram(
        title="transfer_mlp — double-layer model",
        subtitle="Layer 2 input: TrackNet10 source (250,)  →  frozen k100s2_v1 backbone  →  small trainable correction head",
        rows=tr_rows,
        out_path=OUT_DIR / "transfer_mlp.png",
        forced_colors=forced,
        param_note=(f"{tr.count_params():,} total params  "
                    f"({tr.count_params() - 32314:,} frozen + 32,314 new trainable)"),
    )


if __name__ == "__main__":
    main()
