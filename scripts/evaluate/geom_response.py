"""
Dose comparison: midpoint R vs geometric-mean R for Concrete 25 cm.

Old: R sampled at bin midpoints (0.5, 1.5, ..., 249.5 MeV)
New: R = sqrt(R(edge_lo) * R(edge_hi))
     bin i -> edges [i, i+1] MeV; bin 0 uses 1 keV as lower edge

Usage:
    python scripts/evaluate/geom_response.py [--output-dir reports/figures/]
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL, MODEL_PKL_NAME
from shielding_ml.metrics.dose import (
    _build_r_all, _build_r_geom, dose_midpoint, dose_geometric,
)
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_HE, DOSE_TABLE, TRACKNET10_SPECTRUM, REPORTS_DIR,
)

MODELS = [
    ('New CNN', str(REPO_ROOT / '20260713' / 'k25_k11_k3_d256_d64_mae_e25_scale0p1'), 0.1, '#4a90d9'),
]

PHITS_PATH = str(PHITS_HE / 'Concrete_25cm_1.out')

parser = argparse.ArgumentParser(description='Geometric response comparison for Concrete 25 cm')
parser.add_argument('--output-dir', default=None, help='Output directory (default: reports/figures/)')
args = parser.parse_args()

OUT_DIR = args.output_dir if args.output_dir else str(REPORTS_DIR / 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Build response arrays ────────────────────────────────────────────────────
R_MID  = _build_r_all(DOSE_TABLE)   # midpoint response
R_GEOM = _build_r_geom(DOSE_TABLE)  # geometric-mean response

def dose_mid(flux, max_bin=250):
    return dose_midpoint(flux, R_MID, excl_bin0=True, max_bin=max_bin)

def dose_geom(flux, max_bin=250):
    return dose_geometric(flux, R_GEOM, excl_bin0=True, max_bin=max_bin)

def cnn_predict(model, mat, thick, scale):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    return 10**model.predict(X, verbose=0).flatten() * FLUX_SCALE

# ── Load data ─────────────────────────────────────────────────────────────────
A_SPEC = load_spectrum(TRACKNET10_SPECTRUM)
phits_flux, phits_rerr = load_phits(PHITS_PATH)
phits_sig = phits_flux * np.nan_to_num(phits_rerr)

loaded = []
for lbl, folder, scale, color in MODELS:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    print(f'Loading: {lbl}')
    loaded.append((lbl, load_model(pkl_path), scale, color))

cnn_preds = [(lbl, cnn_predict(m, 'Concrete', 25, sc), color)
             for lbl, m, sc, color in loaded]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                         gridspec_kw={'height_ratios': [2.2, 1.6, 1.0],
                                      'hspace': 0.5})
fig.patch.set_facecolor('#f8f9fa')
ax_fl, ax_di, ax_tbl = axes

# ── Flux panel ────────────────────────────────────────────────────────────────
ax_fl.semilogy(E_ALL[1:], phits_flux[1:], color='black', linewidth=1.0,
               alpha=0.5, label='PHITS')
ax_fl.fill_between(E_ALL[1:],
                   np.clip(phits_flux[1:] - phits_sig[1:], 1e-30, None),
                   phits_flux[1:] + phits_sig[1:],
                   color='black', alpha=0.06)
for lbl, pred, color in cnn_preds:
    ax_fl.semilogy(E_ALL[1:], pred[1:], color=color, linewidth=1.4,
                   linestyle='--', label=lbl)
ax_fl.set_xlim(1, 250)
ax_fl.set_ylim(bottom=1e-7)
ax_fl.set_xlabel('Energy (MeV)', fontsize=11)
ax_fl.set_ylabel('Flux  [n/cm2/MeV/src]', fontsize=10)
ax_fl.set_title('Concrete 25 cm — Flux Spectrum', fontsize=11, fontweight='bold', color='#1a252f')
ax_fl.legend(fontsize=9)
ax_fl.grid(True, which='both', alpha=0.2)
ax_fl.set_facecolor('#f0f2f5')

# ── Dose integrand panel ──────────────────────────────────────────────────────
ax_di.semilogy(E_ALL[1:], phits_flux[1:] * R_MID[1:],
               color='black', linewidth=1.2, alpha=0.7, label='PHITS x R (midpoint)')
ax_di.semilogy(E_ALL[1:], phits_flux[1:] * R_GEOM[1:],
               color='black', linewidth=1.2, alpha=0.7, linestyle='--', label='PHITS x R (geom mean)')
for lbl, pred, color in cnn_preds:
    ax_di.semilogy(E_ALL[1:], pred[1:] * R_MID[1:],
                   color=color, linewidth=1.2, linestyle='-', label=f'{lbl} x R (midpoint)')
    ax_di.semilogy(E_ALL[1:], pred[1:] * R_GEOM[1:],
                   color=color, linewidth=1.2, linestyle='--', label=f'{lbl} x R (geom mean)')
ax_di.set_xlim(1, 250)
ax_di.set_xlabel('Energy (MeV)', fontsize=11)
ax_di.set_ylabel('Flux x R  [mrem/h per bin]', fontsize=10)
ax_di.set_title('Dose Integrand per Bin  —  solid: midpoint R   dashed: geom-mean R',
                fontsize=10, fontweight='bold', color='#1a252f')
ax_di.legend(fontsize=8, ncol=2)
ax_di.grid(True, which='both', alpha=0.2)
ax_di.set_facecolor('#f0f2f5')

# ── Dose table ────────────────────────────────────────────────────────────────
ax_tbl.axis('off')

cnn_lbl, cnn_pred, _ = cnn_preds[0]

d_phits_mid   = dose_mid(phits_flux)
d_phits_geom  = dose_geom(phits_flux)
d_cnn_mid     = dose_mid(cnn_pred)
d_cnn_geom    = dose_geom(cnn_pred)

d_phits_mid5  = dose_mid(phits_flux,  max_bin=5)
d_phits_geom5 = dose_geom(phits_flux, max_bin=5)
d_cnn_mid5    = dose_mid(cnn_pred,    max_bin=5)
d_cnn_geom5   = dose_geom(cnn_pred,   max_bin=5)

err_mid   = 100 * (d_cnn_mid   - d_phits_mid)   / d_phits_mid
err_geom  = 100 * (d_cnn_geom  - d_phits_geom)  / d_phits_geom
err_mid5  = 100 * (d_cnn_mid5  - d_phits_mid5)  / d_phits_mid5
err_geom5 = 100 * (d_cnn_geom5 - d_phits_geom5) / d_phits_geom5

def ecol(pct):
    v = abs(pct)
    if v < 20: return (0.55, 0.86, 0.60, 0.75)
    if v < 30: return (1.00, 0.93, 0.45, 0.75)
    return (0.95, 0.45, 0.45, 0.75)

col_labels = ['', 'Midpoint R', 'Geom-mean R']
rows = [
    ['-- 1-250 MeV ----------------', '----------', '----------'],
    ['PHITS dose rate (mrem/h)',       f'{d_phits_mid:.4e}',  f'{d_phits_geom:.4e}'],
    [f'{cnn_lbl} dose rate (mrem/h)', f'{d_cnn_mid:.4e}',    f'{d_cnn_geom:.4e}'],
    ['% err  (CNN - PHITS) / PHITS',  f'{err_mid:+.2f}%',    f'{err_geom:+.2f}%'],
    ['-- 1-5 MeV -----------------',  '----------', '----------'],
    ['PHITS dose rate (mrem/h)',       f'{d_phits_mid5:.4e}', f'{d_phits_geom5:.4e}'],
    [f'{cnn_lbl} dose rate (mrem/h)', f'{d_cnn_mid5:.4e}',   f'{d_cnn_geom5:.4e}'],
    ['% err  (CNN - PHITS) / PHITS',  f'{err_mid5:+.2f}%',   f'{err_geom5:+.2f}%'],
]
_sep  = ['#c8d8e8', '#c8d8e8', '#c8d8e8']
cell_colors = [
    _sep,
    ['#e8f0fe', '#d8eaff', '#d8ffe8'],
    ['#f0f0f0', '#d8eaff', '#d8ffe8'],
    ['#f0f0f0', ecol(err_mid),  ecol(err_geom)],
    _sep,
    ['#e8f0fe', '#d8eaff', '#d8ffe8'],
    ['#f0f0f0', '#d8eaff', '#d8ffe8'],
    ['#f0f0f0', ecol(err_mid5), ecol(err_geom5)],
]

tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   cellColours=cell_colors,
                   loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for j in range(3):
    tbl[0, j].set_facecolor('#1a252f')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, 9):
    tbl[i, 0].set_text_props(ha='left')

fig.suptitle('Dose Integration Method Comparison  —  Concrete 25 cm',
             fontsize=12, fontweight='bold', color='#1a252f', y=1.01)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'geom_response_concrete25cm.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
