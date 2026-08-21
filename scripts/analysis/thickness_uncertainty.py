"""
Single-layer thickness uncertainty analysis.

For each (material, nominal_thickness) combination, sample 1000 thickness
values from N(mu=nominal, sigma=1 cm) and compute CNN-predicted dose rates.

Materials  : Concrete, Steel, BPE  (nominal densities)
Thicknesses: 25, 60, 90 cm
Model      : k100s2 v1 (default)

Outputs
-------
  inference_results/thickness_uncertainty/
    results.csv      — all raw samples
    distributions.pdf  — 3×3 grid: rows=thickness, cols=material
                         histogram of dose + nominal dose marker + stats

Usage
-----
    ../venv/bin/python scripts/analysis/thickness_uncertainty.py
    ../venv/bin/python scripts/analysis/thickness_uncertainty.py --sigma 2
    ../venv/bin/python scripts/analysis/thickness_uncertainty.py --plots-only
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
p.add_argument('--model-dir', default=None)
p.add_argument('--input-scale', type=float, default=0.001)
p.add_argument('--sigma',       type=float, default=1.0,
               help='Std dev of thickness Gaussian in cm (default 1).')
p.add_argument('--n-samples',   type=int,   default=1000)
p.add_argument('--seed',        type=int,   default=42)
p.add_argument('--plots-only',  action='store_true')
args = p.parse_args()

_K100S2_DIR = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
MODEL_DIR = Path(args.model_dir) if args.model_dir else _K100S2_DIR / 'v1'
MODEL_PKL = MODEL_DIR / 'model.pkl'

OUT_DIR = REPO_ROOT / 'inference_results' / 'thickness_uncertainty'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Model : {MODEL_PKL}')
print(f'sigma : {args.sigma} cm   n_samples : {args.n_samples}')
print(f'Output: {OUT_DIR}\n')

# ── constants ─────────────────────────────────────────────────────────────────
MATERIALS = {
    'Concrete': 2.3,
    'Steel':    7.86,
    'BPE':      1.04,
}
NOMINAL_THICKNESSES = [25, 60, 90]
SCALE = args.input_scale

# ── load model ────────────────────────────────────────────────────────────────
if not args.plots_only:
    with open(MODEL_PKL, 'rb') as f:
        model = pickle.load(f)

A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
R_ALL  = _build_r_all(DOSE_TABLE)

def predict_dose(thick: float, dens: float) -> float:
    B  = np.ones(250) * thick * SCALE
    B1 = np.ones(250) * dens  * SCALE
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    flux = 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_midpoint(flux, R_ALL, excl_bin0=True)

# ── sample or load ────────────────────────────────────────────────────────────
CSV_PATH   = OUT_DIR / 'results.csv'
CSV_FIELDS = ['material', 'nominal_thickness_cm', 'sampled_thickness_cm',
              'density_g_cm3', 'dose_pSv_per_src']

rows: list[dict] = []

if args.plots_only and CSV_PATH.exists():
    print(f'Loading {CSV_PATH}')
    with open(CSV_PATH, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'material':             r['material'],
                'nominal_thickness_cm': int(r['nominal_thickness_cm']),
                'sampled_thickness_cm': float(r['sampled_thickness_cm']),
                'density_g_cm3':        float(r['density_g_cm3']),
                'dose_pSv_per_src':     float(r['dose_pSv_per_src']),
            })
else:
    rng   = np.random.default_rng(args.seed)
    total = len(MATERIALS) * len(NOMINAL_THICKNESSES)
    done  = 0
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for mat, dens in MATERIALS.items():
            for nom_thick in NOMINAL_THICKNESSES:
                samples = rng.normal(loc=nom_thick, scale=args.sigma,
                                     size=args.n_samples)
                samples = np.clip(samples, 0.1, None)   # physical lower bound
                for t in samples:
                    dose = predict_dose(t, dens)
                    row  = {
                        'material':             mat,
                        'nominal_thickness_cm': nom_thick,
                        'sampled_thickness_cm': round(float(t), 6),
                        'density_g_cm3':        dens,
                        'dose_pSv_per_src':     dose,
                    }
                    rows.append(row)
                    writer.writerow(row)
                done += 1
                nom_dose = predict_dose(nom_thick, dens)
                print(f'  [{done}/{total}]  {mat} {nom_thick}cm  '
                      f'ρ={dens}  nom_dose={nom_dose:.3e}')
    print(f'\nCSV → {CSV_PATH}')

# ── restructure ───────────────────────────────────────────────────────────────
# data[mat][nom_thick] = (sampled_thicknesses, doses)
data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
for mat in MATERIALS:
    data[mat] = {}
    for nom in NOMINAL_THICKNESSES:
        sub = [r for r in rows
               if r['material'] == mat and r['nominal_thickness_cm'] == nom]
        data[mat][nom] = (
            np.array([r['sampled_thickness_cm'] for r in sub]),
            np.array([r['dose_pSv_per_src']     for r in sub]),
        )

# ── nominal doses (at exact nominal thickness) ────────────────────────────────
# Load model if plots-only (needed for nominal dose lines)
if args.plots_only:
    with open(MODEL_PKL, 'rb') as f:
        model = pickle.load(f)

nom_doses: dict[str, dict[int, float]] = {}
for mat, dens in MATERIALS.items():
    nom_doses[mat] = {t: predict_dose(t, dens) for t in NOMINAL_THICKNESSES}

# ── colour scheme ─────────────────────────────────────────────────────────────
MAT_COLORS = {'Concrete': '#4e79a7', 'Steel': '#e15759', 'BPE': '#59a14f'}
THICK_LABELS = {25: '25 cm', 60: '60 cm', 90: '90 cm'}

# ── pre-compute Gaussian fits for all (mat, nom) — used in plots + table ──────
fits: dict[str, dict[int, dict]] = {}
for mat, dens in MATERIALS.items():
    fits[mat] = {}
    for nom in NOMINAL_THICKNESSES:
        thicks, doses = data[mat][nom]
        mu_fit, sigma_fit = sp_norm.fit(doses)
        fits[mat][nom] = {
            'mu':       mu_fit,
            'sigma':    sigma_fit,
            'cv_pct':   100 * sigma_fit / mu_fit,
            'nom_dose': nom_doses[mat][nom],
        }

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
DIST_PDF = OUT_DIR / 'distributions.pdf'
with PdfPages(DIST_PDF) as pdf:

    # ── Page 1: 3×4 grid — col 0 = input thickness dist, cols 1-3 = dose ──────
    fig, axes = plt.subplots(3, 4, figsize=(18, 12),
                             gridspec_kw={'width_ratios': [1, 1.2, 1.2, 1.2]})
    fig.suptitle(
        f'Thickness uncertainty → dose distribution  '
        f'(σ_t = {args.sigma} cm, N = {args.n_samples},  model: {MODEL_DIR.name})',
        fontsize=12, fontweight='bold',
    )

    for row_i, nom in enumerate(NOMINAL_THICKNESSES):
        # ── column 0: input thickness Gaussian ──────────────────────────────
        ax_t  = axes[row_i, 0]
        thicks_any, _ = data['Concrete'][nom]   # same samples for all materials
        cnt_t, edges_t, _ = ax_t.hist(
            thicks_any, bins=30, color='#aac4e0', alpha=0.8,
            edgecolor='white', linewidth=0.4, label='samples',
        )
        # overlay fitted Gaussian
        mu_t, sig_t = sp_norm.fit(thicks_any)
        x_t  = np.linspace(thicks_any.min(), thicks_any.max(), 300)
        bw_t = edges_t[1] - edges_t[0]
        ax_t.plot(x_t, sp_norm.pdf(x_t, mu_t, sig_t) * len(thicks_any) * bw_t,
                  color='#1a3a5c', lw=1.8, label=f'N({mu_t:.1f}, {sig_t:.2f}²)')
        ax_t.axvline(nom,              color='k',    lw=1.5, ls='--', label=f't₀={nom} cm')
        ax_t.axvline(nom - args.sigma, color='gray', lw=1,   ls=':')
        ax_t.axvline(nom + args.sigma, color='gray', lw=1,   ls=':', label=f'±{args.sigma} cm')
        ax_t.set_title(f'Input: N({nom}, {args.sigma}²) cm', fontsize=9, fontweight='bold')
        ax_t.set_xlabel('Thickness (cm)', fontsize=8)
        ax_t.set_ylabel('Count', fontsize=8)
        ax_t.tick_params(labelsize=7)
        ax_t.legend(fontsize=6.5)

        # ── columns 1-3: dose distribution per material ──────────────────────
        for col_j, (mat, dens) in enumerate(MATERIALS.items()):
            ax    = axes[row_i, col_j + 1]
            color = MAT_COLORS[mat]
            thicks, doses = data[mat][nom]
            f_    = fits[mat][nom]
            nom_d = f_['nom_dose']
            mu_d, sig_d, cv = f_['mu'], f_['sigma'], f_['cv_pct']

            cnt, edges, _ = ax.hist(
                doses, bins=40, color=color, alpha=0.65,
                edgecolor='white', linewidth=0.4, label='samples',
            )
            ax.axvline(nom_d, color='k', lw=1.6, ls='--',
                       label=f'nominal  {nom_d:.3e}')

            x_d  = np.linspace(doses.min(), doses.max(), 300)
            bw_d = edges[1] - edges[0]
            ax.plot(x_d, sp_norm.pdf(x_d, mu_d, sig_d) * len(doses) * bw_d,
                    color='#1a1a1a', lw=1.8,
                    label=f'fit  μ={mu_d:.3e}\n      σ={sig_d:.2e}')

            ax.set_title(
                f'{mat}  (ρ={dens} g/cm³)\nCV = {cv:.2f}%',
                fontsize=8, fontweight='bold',
            )
            ax.set_xlabel('Dose (pSv / source n)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc='upper right')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: dose vs sampled thickness scatter ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Dose vs sampled thickness  (σ_t = {args.sigma} cm)',
                 fontsize=12, fontweight='bold')

    for ax, (mat, dens) in zip(axes, MATERIALS.items()):
        color = MAT_COLORS[mat]
        for nom in NOMINAL_THICKNESSES:
            thicks, doses = data[mat][nom]
            ax.scatter(thicks, doses, s=4, alpha=0.35, color=color,
                       label=f'{nom} cm')
            ax.scatter([nom], [nom_doses[mat][nom]],
                       s=80, color='k', marker='D', zorder=5)
        ax.set_yscale('log')
        ax.set_xlabel('Sampled thickness (cm)', fontsize=9)
        ax.set_ylabel('Dose (pSv / source n)', fontsize=9)
        ax.set_title(f'{mat}  (ρ = {dens} g/cm³)', fontsize=10)
        ax.legend(fontsize=8, title='nominal')
        ax.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 3: summary table ─────────────────────────────────────────────────
    col_labels = ['Material', 't₀ (cm)', 'Fit μ (pSv/src)', 'Fit σ (pSv/src)',
                  'CV (%)', 'σ < 10% μ']
    table_rows = []
    row_colors = []
    _mat_bg = {'Concrete': '#ddeeff', 'Steel': '#fde8e8', 'BPE': '#e4f4e4'}

    for mat, dens in MATERIALS.items():
        for nom in NOMINAL_THICKNESSES:
            f_ = fits[mat][nom]
            cv  = f_['cv_pct']
            ok  = cv < 10.0
            table_rows.append([
                mat,
                str(nom),
                f'{f_["mu"]:.4e}',
                f'{f_["sigma"]:.4e}',
                f'{cv:.2f}%',
                '✓' if ok else '✗',
            ])
            flag_col = '#c8f7c5' if ok else '#ffcdd2'
            row_colors.append([_mat_bg[mat]] * 4 + [flag_col, flag_col])

    fig, ax_tbl = plt.subplots(figsize=(12, 5))
    ax_tbl.axis('off')
    fig.suptitle(
        f'Summary: Gaussian fit to dose distributions\n'
        f'σ_t = {args.sigma} cm,  N = {args.n_samples},  model: {MODEL_DIR.name}',
        fontsize=12, fontweight='bold', y=0.97,
    )

    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellColours=row_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    fig.text(0.13, 0.01, '✓ = σ/μ < 10%  (green)     ✗ = σ/μ ≥ 10%  (red)',
             fontsize=8, color='#333333')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f'PDF → {DIST_PDF}')
print('Done.')
