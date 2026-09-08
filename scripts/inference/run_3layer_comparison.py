"""
Compare all four 3-layer chaining approaches (scripts/inference/three_layer.py)
against a PHITS 3-layer reference.

Auto-discovers every .out file in data/raw/phits/3layer/ and runs all four
approaches against each. Saves a comparison PDF (one page per case) plus
prints a console summary.

Usage:
    ../venv/bin/python scripts/inference/run_3layer_comparison.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shielding_ml.data.constants import E_ALL, MATERIALS
from shielding_ml.data.loaders import load_phits
from shielding_ml.pipelines.paths import PHITS_3L, REPO_ROOT

from scripts.inference.single_layer import load_model, compute_dose
from scripts.inference.three_layer  import predict_3layer_all_approaches, APPROACH_LABELS

BIN_SLICE = slice(20, 150)
COLORS = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]   # one per approach, in APPROACH_LABELS order

OUT_DIR = REPO_ROOT / "inference_results" / "three_layer_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "comparison.pdf"


def avg_ratio(pred: np.ndarray, phits: np.ndarray) -> float:
    mask = phits[BIN_SLICE] > 0
    return float(np.mean(pred[BIN_SLICE][mask] / phits[BIN_SLICE][mask]))


def mape(pred: np.ndarray, phits: np.ndarray) -> float:
    mask = phits[BIN_SLICE] > 0
    return float(np.mean(np.abs(pred[BIN_SLICE][mask] - phits[BIN_SLICE][mask])
                          / phits[BIN_SLICE][mask]) * 100)


def _err_color(pct: float):
    v = abs(pct)
    if v < 20:  return (0.55, 0.86, 0.60, 0.7)
    if v < 50:  return (1.00, 0.93, 0.45, 0.7)
    return (0.95, 0.45, 0.45, 0.7)


def parse_3layer_name(path: Path):
    pairs = re.findall(r"([A-Za-z]+)_(\d+)cm", path.stem)
    if len(pairs) != 3:
        return None
    return [(m, int(t)) for m, t in pairs]


def _render_single_approach_page(pdf, case_lbl, phits_flux, phits_dose, pred, label, color):
    """One-approach detail page: flux overlay, ratio, flux error, single-row stats."""
    ar  = avg_ratio(pred, phits_flux)
    mp  = mape(pred, phits_flux)
    d   = compute_dose(pred)
    pct = 100 * (d - phits_dose) / phits_dose

    fig = plt.figure(figsize=(14, 12))
    gs  = plt.GridSpec(3, 2, height_ratios=[2.2, 1.3, 1.3], hspace=0.55, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    ax1.semilogy(E_ALL, phits_flux, "k-", lw=2, label="PHITS", zorder=5)
    ax1.semilogy(E_ALL, pred, color=color, lw=1.6, ls="--", label=label, zorder=4)
    ax1.set_xlim(0, 250); ax1.set_ylim(bottom=1e-7)
    ax1.set_xlabel("Energy (MeV)"); ax1.set_ylabel("Flux [n/cm²/source]")
    ax1.set_title(case_lbl, fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")

    ratio = np.where(phits_flux > 0, pred / phits_flux, np.nan)
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.axhspan(0.8, 1.2, alpha=0.08, color="green")
    ax2.plot(E_ALL[BIN_SLICE], ratio[BIN_SLICE], color=color, lw=1.4)
    ax2.set_xlim(20, 150)
    ax2.set_xlabel("Energy (MeV)"); ax2.set_ylabel("CNN / PHITS")
    ax2.set_title("Ratio (bins 20–150)")
    finite = ratio[BIN_SLICE][np.isfinite(ratio[BIN_SLICE])]
    if finite.size:
        ax2.set_ylim(0, min(np.nanpercentile(finite, 98) * 1.2, 15))

    flux_err = np.where(phits_flux > 0, (pred - phits_flux) / phits_flux * 100, np.nan)
    ax3.axhline(0.0, color="k", ls="--", lw=1)
    ax3.axhspan(-20, 20, alpha=0.08, color="green")
    ax3.plot(E_ALL[BIN_SLICE], flux_err[BIN_SLICE], color=color, lw=1.4)
    ax3.set_xlim(20, 150)
    ax3.set_xlabel("Energy (MeV)"); ax3.set_ylabel("Flux error [%]")
    ax3.set_title("Flux error, (CNN − PHITS) / PHITS × 100  (bins 20–150)")
    err_finite = flux_err[BIN_SLICE][np.isfinite(flux_err[BIN_SLICE])]
    if err_finite.size:
        lo, hi = np.nanpercentile(err_finite, [2, 98])
        pad = 0.15 * max(hi - lo, 1.0)
        ax3.set_ylim(lo - pad, hi + pad)

    stats_text = (
        f"avg_ratio = {ar:.4f}      MAPE = {mp:.1f}%      "
        f"CNN dose = {d:.3e} mrem/hr      PHITS dose = {phits_dose:.3e} mrem/hr      "
        f"Dose error = {pct:+.1f}%"
    )
    ax_stats.text(0.5, 0.6, stats_text, ha="center", va="center", fontsize=11,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=_err_color(pct),
                            edgecolor="#999999"))

    fig.suptitle(f"{label}  |  {case_lbl}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cases = sorted(PHITS_3L.glob("*.out"))
    if not cases:
        print(f"No .out files found in {PHITS_3L}")
        return 1

    v1 = load_model(REPO_ROOT / "models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/primary/k100s2_v1/model.pkl")
    tr = load_model(REPO_ROOT / "models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/primary/transfer_mlp/transfer_model.pkl")

    labels = list(APPROACH_LABELS.values())

    print(f"\n{'Case':<45} {'Approach':<55} {'avg_ratio':>10} {'MAPE%':>8} {'dose CNN':>12} {'dose err%':>10}")
    print("-" * 145)

    with PdfPages(OUT_PDF) as pdf:
        for path in cases:
            layers = parse_3layer_name(path)
            if not layers or any(m not in MATERIALS for m, _ in layers):
                print(f"Skipping {path.name} — couldn't parse 3 (material, thickness) pairs")
                continue
            (mat1, t1), (mat2, t2), (mat3, t3) = layers

            phits_flux, _ = load_phits(str(path))
            if phits_flux is None:
                continue
            phits_dose = compute_dose(phits_flux)
            case_lbl = path.stem.replace("_", " ")

            preds = predict_3layer_all_approaches(v1, tr, mat1, t1, mat2, t2, mat3, t3)

            fig = plt.figure(figsize=(14, 13))
            gs  = plt.GridSpec(3, 2, height_ratios=[2.2, 1.3, 1.3], hspace=0.55, figure=fig)
            ax1    = fig.add_subplot(gs[0, 0])
            ax2    = fig.add_subplot(gs[0, 1])
            ax3    = fig.add_subplot(gs[1, :])
            ax_tbl = fig.add_subplot(gs[2, :])
            ax_tbl.axis("off")

            ax1.semilogy(E_ALL, phits_flux, "k-", lw=2, label="PHITS", zorder=5)
            ax2.axhline(1.0, color="k", ls="--", lw=1)
            ax2.axhspan(0.8, 1.2, alpha=0.08, color="green")
            ax3.axhline(0.0, color="k", ls="--", lw=1)
            ax3.axhspan(-20, 20, alpha=0.08, color="green")

            dose_rows = []
            ratios_for_ylim = []
            errs_for_ylim = []
            for label, color in zip(labels, COLORS):
                pred = preds[label]
                ar  = avg_ratio(pred, phits_flux)
                mp  = mape(pred, phits_flux)
                d   = compute_dose(pred)
                pct = 100 * (d - phits_dose) / phits_dose
                dose_rows.append((label, ar, mp, d, pct, color))
                print(f"{case_lbl:<45} {label:<55} {ar:>10.4f} {mp:>8.2f} {d:>12.3e} {pct:>+8.1f}%")

                ax1.semilogy(E_ALL, pred, color=color, lw=1.4, ls="--",
                             label=f"{label}  (ratio={ar:.2f})")

                ratio = np.where(phits_flux > 0, pred / phits_flux, np.nan)
                ax2.plot(E_ALL[BIN_SLICE], ratio[BIN_SLICE], color=color, lw=1.2, label=label)
                ratios_for_ylim.append(ratio[BIN_SLICE])

                flux_err = np.where(phits_flux > 0, (pred - phits_flux) / phits_flux * 100, np.nan)
                ax3.plot(E_ALL[BIN_SLICE], flux_err[BIN_SLICE], color=color, lw=1.2, label=label)
                errs_for_ylim.append(flux_err[BIN_SLICE])

            ax1.set_xlim(0, 250); ax1.set_ylim(bottom=1e-7)
            ax1.set_xlabel("Energy (MeV)"); ax1.set_ylabel("Flux [n/cm²/source]")
            ax1.set_title(case_lbl, fontsize=10, fontweight="bold")
            ax1.legend(fontsize=7, loc="upper right")

            ax2.set_xlim(20, 150)
            ax2.set_xlabel("Energy (MeV)"); ax2.set_ylabel("CNN / PHITS")
            ax2.set_title("Ratio (bins 20–150)")
            ax2.legend(fontsize=7)
            valid = np.concatenate(ratios_for_ylim)
            finite = valid[np.isfinite(valid)]
            if finite.size:
                ax2.set_ylim(0, min(np.nanpercentile(finite, 98) * 1.2, 15))

            ax3.set_xlim(20, 150)
            ax3.set_xlabel("Energy (MeV)"); ax3.set_ylabel("Flux error [%]")
            ax3.set_title("Flux error, (CNN − PHITS) / PHITS × 100  (bins 20–150)")
            ax3.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
            err_valid = np.concatenate(errs_for_ylim)
            err_finite = err_valid[np.isfinite(err_valid)]
            if err_finite.size:
                lo, hi = np.nanpercentile(err_finite, [2, 98])
                pad = 0.15 * max(hi - lo, 1.0)
                ax3.set_ylim(lo - pad, hi + pad)

            col_labels = ["Approach", "avg_ratio", "MAPE %", "CNN dose (mrem/hr)",
                          "Dose error %", "PHITS dose (mrem/hr)"]
            tbl_data = [[n, f"{ar:.4f}", f"{mp:.1f}%", f"{d:.3e}", f"{pct:+.1f}%", f"{phits_dose:.3e}"]
                        for n, ar, mp, d, pct, _ in dose_rows]
            cell_colors = [["white", "white", "white", "white", _err_color(pct), "#d0e8ff"]
                           for _, _, _, _, pct, _ in dose_rows]

            tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                                cellColours=cell_colors, loc="center", cellLoc="center",
                                colWidths=[0.34, 0.11, 0.10, 0.16, 0.13, 0.16])
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
            for i in range(len(tbl_data)):
                tbl[i + 1, 0].set_text_props(ha="left")
                tbl[i + 1, 0].PAD = 0.02
            for j in range(len(col_labels)):
                tbl[0, j].set_facecolor("#2c3e50")
                tbl[0, j].set_text_props(color="white", fontweight="bold")

            fig.suptitle(f"Three-layer chaining approaches  |  {case_lbl}",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── page 2: approach 4 alone (actual output chained through both
            # extra layers — no TrackNet10 refeed at either hop) ────────────
            approach_4_label = APPROACH_LABELS[(4, "actual", "actual")]
            _render_single_approach_page(
                pdf, case_lbl, phits_flux, phits_dose,
                pred=preds[approach_4_label], label=approach_4_label, color=COLORS[3],
            )
            print()

    print(f"Saved → {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
