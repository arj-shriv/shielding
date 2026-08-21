"""
Generate images_for_paper/multilayer/<Mat1>_<X>cm_<Mat2>_<Y>cm.png

One clean flux plot per multilayer PHITS case using the TrackNet10-source
layer-2 approach (best performing): v1 for layer-1 amplitude, transfer_cnn_source_mlp
for layer 2 fed with TrackNet10 as source.

  - PHITS (black) vs CNN chained (colour)
  - Dose error annotated below plot
  - No ratio subplot, no table
"""

from __future__ import annotations
import re, sys, pickle, os
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import FLUX_SCALE, MATERIALS, E_ALL
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_ML, DATA_REF, TRACKNET10_SPECTRUM, DOSE_TABLE,
)

OUT   = Path(__file__).parent / 'multilayer'
OUT.mkdir(parents=True, exist_ok=True)
SCALE     = 0.001
BIN_SLICE = slice(20, 150)

# ── load models + helpers ─────────────────────────────────────────────────────
MODEL_BASE = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'

def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_v1  = _load(MODEL_BASE / 'v1' / 'model.pkl')
model_cnn = _load(MODEL_BASE / 'transfer_cnn_source_mlp' / 'transfer_model.pkl')

A_SPEC    = load_spectrum(str(TRACKNET10_SPECTRUM))
VOID_FLUX, _ = load_phits(str(DATA_REF / 'No_Shielding.out'))
R_ALL     = _build_r_all(DOSE_TABLE)

def _predict(mdl, src, mat, thick):
    dens = MATERIALS[mat]
    B  = np.ones(250) * thick * SCALE
    B1 = np.ones(250) * dens  * SCALE
    X  = np.array([src, B, B1]).T.reshape(1, 250, 3)
    return 10 ** mdl.predict(X, verbose=0).flatten() * FLUX_SCALE

def chained_flux(mat1, thick1, mat2, thick2):
    pred_L1 = _predict(model_v1, A_SPEC, mat1, thick1)
    a       = pred_L1[BIN_SLICE].sum() / VOID_FLUX[BIN_SLICE].sum()
    pred_L2 = _predict(model_cnn, A_SPEC, mat2, thick2)
    return pred_L2 * a

# ── colour: first material determines colour ───────────────────────────────────
MAT_COLOR = {'Concrete': '#4e79a7', 'Steel': '#e15759', 'BPE': '#59a14f'}

# ── discover multilayer PHITS cases ──────────────────────────────────────────
def parse_layers(path):
    name = os.path.basename(path).replace('.out', '')
    return [(m, int(t)) for m, t in re.findall(r'([A-Za-z]+)_(\d+)cm', name)]

cases = sorted(PHITS_ML.glob('*.out'))

# ── plot each case ────────────────────────────────────────────────────────────
for p in cases:
    layers = parse_layers(str(p))
    if not layers or any(m not in MATERIALS for m, _ in layers):
        continue

    phits_flux, _ = load_phits(str(p))
    if phits_flux is None:
        continue

    (mat1, thick1), (mat2, thick2) = layers[0], layers[1]
    pred = chained_flux(mat1, thick1, mat2, thick2)

    phits_dose = dose_midpoint(phits_flux, R_ALL, excl_bin0=True)
    cnn_dose   = dose_midpoint(pred,       R_ALL, excl_bin0=True)
    pct_err    = 100 * (cnn_dose - phits_dose) / phits_dose
    color      = MAT_COLOR.get(mat1, '#333333')

    case_lbl = f'{mat1} {thick1} cm  +  {mat2} {thick2} cm'

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.semilogy(E_ALL, phits_flux, color='black', lw=2.0, label='PHITS', zorder=5)
    ax.semilogy(E_ALL, pred,       color=color,   lw=1.8, ls='--',
                label='CNN model', zorder=4)

    ax.set_xlim(0, 250)
    ax.set_ylim(bottom=1e-7)
    ax.set_xlabel('Neutron energy (MeV)', fontsize=11)
    ax.set_ylabel('Flux  [n / cm² / source n]', fontsize=11)
    ax.set_title(case_lbl, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.22)

    sign = '+' if pct_err >= 0 else ''
    fig.text(0.5, -0.02,
             f'Dose error:  {sign}{pct_err:.1f}%   '
             f'(CNN {cnn_dose:.3e}  vs  PHITS {phits_dose:.3e}  mrem/hr)',
             ha='center', fontsize=9, color='#333333')

    plt.tight_layout()
    fname = OUT / f'{mat1}_{thick1}cm_{mat2}_{thick2}cm.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {fname.name}   dose err {sign}{pct_err:.1f}%')

print(f'\nDone → {OUT}')
