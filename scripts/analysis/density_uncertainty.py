"""
Single-layer density uncertainty analysis.

For each (material, thickness) combination, vary density ±10% around the
nominal value using 50 evenly-spaced samples (25 below, 25 above) and record
the CNN-predicted dose rate.

Outputs
-------
  inference_results/density_uncertainty/
    results.csv          — all raw results
    scatter.pdf          — dose vs density offset%, one subplot per material
    histograms.pdf       — dose histogram per (material, thickness)

Usage
-----
    ../venv/bin/python scripts/analysis/density_uncertainty.py
    ../venv/bin/python scripts/analysis/density_uncertainty.py \\
        --model-dir models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/v1
"""

from __future__ import annotations
import argparse, pickle, sys, csv
from pathlib import Path

import numpy as np
from scipy.stats import norm as sp_norm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_spectrum
from shielding_ml.pipelines.paths import REPO_ROOT, TRACKNET10_SPECTRUM, DOSE_TABLE
from shielding_ml.data.constants import FLUX_SCALE
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--model-dir', default=None,
               help='Path to model dir containing model.pkl. '
                    'Defaults to k100s2.../v1/')
p.add_argument('--input-scale', type=float, default=0.001)
p.add_argument('--n-samples',   type=int,   default=50,
               help='Total density samples (half below, half above nominal).')
p.add_argument('--pct-range',   type=float, default=10.0,
               help='Density variation in percent (default ±10%).')
p.add_argument('--plots-only',  action='store_true',
               help='Skip inference; read existing results.csv and replot.')
args = p.parse_args()

_K100S2_DIR = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
MODEL_DIR = Path(args.model_dir) if args.model_dir else _K100S2_DIR / 'v1'
MODEL_PKL = MODEL_DIR / 'model.pkl'

OUT_DIR = REPO_ROOT / 'inference_results' / 'density_uncertainty'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Model : {MODEL_PKL}')
print(f'Output: {OUT_DIR}\n')

# ── constants ─────────────────────────────────────────────────────────────────
MATERIALS = {
    'Concrete': 2.3,
    'Steel':    7.86,
    'BPE':      1.04,
}
THICKNESSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]

SCALE     = args.input_scale
N         = args.n_samples
PCT       = args.pct_range / 100.0          # fraction, e.g. 0.10

# 50 offsets: 25 negative, 25 positive; symmetric about 0
_half = N // 2
OFFSETS = np.concatenate([
    np.linspace(-PCT, 0, _half + 1)[:-1],   # -10% … just below 0
    np.linspace(0,  PCT, _half + 1)[1:],    # just above 0 … +10%
])                                           # shape (N,)

# ── load model and helpers ────────────────────────────────────────────────────
with open(MODEL_PKL, 'rb') as f:
    model = pickle.load(f)

A_SPEC  = load_spectrum(str(TRACKNET10_SPECTRUM))
R_ALL   = _build_r_all(DOSE_TABLE)

def predict_dose(thick: float, dens: float) -> float:
    B  = np.ones(250) * thick * SCALE
    B1 = np.ones(250) * dens  * SCALE
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    flux = 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_midpoint(flux, R_ALL, excl_bin0=True)

# ── run or load ───────────────────────────────────────────────────────────────
CSV_PATH = OUT_DIR / 'results.csv'
fieldnames = [
    'material', 'thickness_cm', 'nominal_density_g_cm3',
    'density_g_cm3', 'density_offset_pct', 'dose_pSv_per_src',
]

rows: list[dict] = []

if args.plots_only and CSV_PATH.exists():
    print(f'Loading existing CSV → {CSV_PATH}')
    with open(CSV_PATH, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'material':              row['material'],
                'thickness_cm':          int(row['thickness_cm']),
                'nominal_density_g_cm3': float(row['nominal_density_g_cm3']),
                'density_g_cm3':         float(row['density_g_cm3']),
                'density_offset_pct':    float(row['density_offset_pct']),
                'dose_pSv_per_src':      float(row['dose_pSv_per_src']),
            })
else:
    total = len(MATERIALS) * len(THICKNESSES) * N
    done  = 0
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mat, nom_dens in MATERIALS.items():
            for thick in THICKNESSES:
                nom_dose = predict_dose(thick, nom_dens)
                for offset in OFFSETS:
                    dens = nom_dens * (1 + offset)
                    dose = predict_dose(thick, dens)
                    row = {
                        'material':               mat,
                        'thickness_cm':           thick,
                        'nominal_density_g_cm3':  round(nom_dens, 4),
                        'density_g_cm3':          round(dens,     6),
                        'density_offset_pct':     round(offset * 100, 4),
                        'dose_pSv_per_src':       dose,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    done += 1
                print(f'  [{done}/{total}]  {mat} {thick}cm  '
                      f'nom_dens={nom_dens}  nom_dose={nom_dose:.3e}')
    print(f'\nCSV saved → {CSV_PATH}')

# ── restructure for plotting ──────────────────────────────────────────────────
# data[mat][thick] = (offsets_array, doses_array)
data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
for mat in MATERIALS:
    data[mat] = {}
    for thick in THICKNESSES:
        subset = [r for r in rows
                  if r['material'] == mat and r['thickness_cm'] == thick]
        data[mat][thick] = (
            np.array([r['density_offset_pct']  for r in subset]),
            np.array([r['dose_pSv_per_src']    for r in subset]),
            np.array([r['density_g_cm3']        for r in subset]),
        )

# ── colour helpers ────────────────────────────────────────────────────────────
_cmap     = plt.colormaps['plasma'].resampled(len(THICKNESSES))
THICKNESS_COLORS = {t: _cmap(i) for i, t in enumerate(THICKNESSES)}

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Scatter: dose vs density offset %  (one subplot per material)
# ─────────────────────────────────────────────────────────────────────────────
SCATTER_PDF = OUT_DIR / 'scatter.pdf'
with PdfPages(SCATTER_PDF) as pdf:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        f'Single-layer dose vs density offset  '
        f'(±{args.pct_range:.0f}%,  {N} samples,  '
        f'model: {MODEL_DIR.name})',
        fontsize=11, fontweight='bold',
    )

    for ax, (mat, nom_dens) in zip(axes, MATERIALS.items()):
        for thick in THICKNESSES:
            offsets, doses, densities = data[mat][thick]
            ax.semilogy(offsets, doses,
                        color=THICKNESS_COLORS[thick],
                        lw=1.4, marker='o', markersize=2,
                        label=f'{thick} cm')

        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('Density offset (%)')
        ax.set_ylabel('Dose (pSv / source n)')
        ax.set_title(f'{mat}  (ρ₀ = {nom_dens} g/cm³)')
        ax.legend(fontsize=7, ncol=2, title='thickness')
        ax.grid(True, which='both', alpha=0.25)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Second page: dose vs absolute density (easier to read for non-experts)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('Dose vs absolute density', fontsize=11, fontweight='bold')

    for ax, (mat, nom_dens) in zip(axes, MATERIALS.items()):
        for thick in THICKNESSES:
            offsets, doses, densities = data[mat][thick]
            ax.semilogy(densities, doses,
                        color=THICKNESS_COLORS[thick],
                        lw=1.4, marker='o', markersize=2,
                        label=f'{thick} cm')

        ax.axvline(nom_dens, color='k', lw=0.8, ls='--', alpha=0.5,
                   label=f'nominal ({nom_dens})')
        ax.set_xlabel('Density (g/cm³)')
        ax.set_ylabel('Dose (pSv / source n)')
        ax.set_title(f'{mat}  (ρ₀ = {nom_dens} g/cm³)')
        ax.legend(fontsize=7, ncol=2, title='thickness')
        ax.grid(True, which='both', alpha=0.25)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f'Scatter PDF saved → {SCATTER_PDF}')

# ── pre-compute Gaussian fits for all (material, thickness) ──────────────────
_all_fits: dict[str, dict[int, tuple[float, float]]] = {}
for _mat in MATERIALS:
    _all_fits[_mat] = {}
    for _thick in THICKNESSES:
        _, _doses, _ = data[_mat][_thick]
        _all_fits[_mat][_thick] = sp_norm.fit(_doses)   # (mu, sigma)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Histograms: dose distribution per (material, thickness)
# ─────────────────────────────────────────────────────────────────────────────
HIST_PDF = OUT_DIR / 'histograms.pdf'
with PdfPages(HIST_PDF) as pdf:
    for mat, nom_dens in MATERIALS.items():
        fig, axes = plt.subplots(4, 3, figsize=(14, 16))
        fig.suptitle(
            f'{mat}  —  dose distribution over ±{args.pct_range:.0f}% density variation\n'
            f'nominal ρ = {nom_dens} g/cm³  |  {N} samples  |  '
            f'model: {MODEL_DIR.name}',
            fontsize=11, fontweight='bold',
        )

        for ax, thick in zip(axes.flat, THICKNESSES):
            offsets, doses, densities = data[mat][thick]
            nom_dose = predict_dose(thick, nom_dens)

            counts, edges, _ = ax.hist(
                doses, bins=15, color=THICKNESS_COLORS[thick],
                alpha=0.7, edgecolor='white', linewidth=0.5, label='samples',
            )
            ax.axvline(nom_dose, color='k', lw=1.5, ls='--',
                       label=f'ρ₀  {nom_dose:.2e}')

            mu_fit, sig_fit = sp_norm.fit(doses)
            x_fit = np.linspace(doses.min(), doses.max(), 300)
            bw    = edges[1] - edges[0]
            ax.plot(x_fit, sp_norm.pdf(x_fit, mu_fit, sig_fit) * len(doses) * bw,
                    color='#1a1a1a', lw=1.6,
                    label=f'fit μ={mu_fit:.2e}\n    σ={sig_fit:.2e}')

            cv = 100 * sig_fit / mu_fit
            ax.set_title(
                f'{thick} cm   ρ: {densities.min():.3f}–{densities.max():.3f} g/cm³\n'
                f'CV = {cv:.2f}%',
                fontsize=8,
            )
            ax.set_xlabel('Dose (pSv / source n)', fontsize=7)
            ax.set_ylabel('Count', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc='upper right')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ── Final page: summary table of Gaussian fit parameters ─────────────────
    _mat_bg = {'Concrete': '#ddeeff', 'Steel': '#fde8e8', 'BPE': '#e4f4e4'}
    col_labels = ['Material', 'Thickness\n(cm)', 'μ (pSv/src)',
                  'σ (pSv/src)', 'CV (%)', 'σ < 10% μ']
    tbl_data, tbl_colors = [], []
    for mat, nom_dens in MATERIALS.items():
        for thick in THICKNESSES:
            mu_f, sig_f = _all_fits[mat][thick]
            cv = 100 * sig_f / mu_f
            ok = cv < 10.0
            tbl_data.append([
                mat, str(thick),
                f'{mu_f:.3e}', f'{sig_f:.3e}',
                f'{cv:.2f}%',
                '✓' if ok else '✗',
            ])
            flag_col = '#c8f7c5' if ok else '#ffcdd2'
            tbl_colors.append([_mat_bg[mat], _mat_bg[mat],
                                'white', 'white', flag_col, flag_col])

    n_rows = len(tbl_data)
    fig, ax_tbl = plt.subplots(figsize=(13, 0.38 * n_rows + 2.5))
    ax_tbl.axis('off')
    fig.suptitle(
        f'Summary: dose rate Gaussian fit parameters\n'
        f'±{args.pct_range:.0f}% density variation,  {N} samples  |  '
        f'model: {MODEL_DIR.name}',
        fontsize=11, fontweight='bold', y=0.97,
    )
    tbl = ax_tbl.table(
        cellText=tbl_data, colLabels=col_labels,
        cellColours=tbl_colors, loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # legend
    fig.text(0.13, 0.01, '✓ = σ/μ < 10%  (green)     ✗ = σ/μ ≥ 10%  (red)',
             fontsize=8, color='#333333')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f'Histogram PDF saved → {HIST_PDF}')
print('\nDone.')
