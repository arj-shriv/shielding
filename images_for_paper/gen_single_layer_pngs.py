"""
Generate images_for_paper/single_layer/<Material>_<X>cm.png

One clean flux plot per PHITS single-layer case:
  - PHITS (black) vs k100s2 v1 CNN (colour)
  - Dose error annotated below plot
  - No ratio subplot, no table
"""

from __future__ import annotations
import re, sys, pickle
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
    REPO_ROOT, PHITS_HE, TRACKNET10_SPECTRUM, DOSE_TABLE,
)

OUT   = Path(__file__).parent / 'single_layer'
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 0.001

MAT_COLOR = {'Concrete': '#4e79a7', 'Steel': '#e15759', 'BPE': '#59a14f'}

# ── load model + helpers ──────────────────────────────────────────────────────
MODEL_PKL = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240' / 'v1' / 'model.pkl'
with open(MODEL_PKL, 'rb') as f:
    model = pickle.load(f)

A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
R_ALL  = _build_r_all(DOSE_TABLE)

def cnn_flux(mat, thick):
    dens = MATERIALS[mat]
    B  = np.ones(250) * thick * SCALE
    B1 = np.ones(250) * dens  * SCALE
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    return 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE

# ── discover PHITS cases (same logic as run_all_inference.py) ─────────────────
candidates: dict[tuple[str, int], Path] = {}
for f in sorted(Path(PHITS_HE).iterdir()):
    if f.suffix not in {'.out', '.txt', ''} or f.name.startswith('.'):
        continue
    if re.search(r'_2$', f.stem):
        continue
    m = re.match(r'^([A-Za-z]+)_(\d+)cm', f.stem)
    if not m:
        continue
    mat, thick = m.group(1), int(m.group(2))
    if mat not in MATERIALS:
        continue
    is_v1 = bool(re.search(r'_1$', f.stem))
    key = (mat, thick)
    if key not in candidates or is_v1:
        candidates[key] = f

cases = sorted(candidates.items(), key=lambda x: (x[0][0], x[0][1]))

# ── plot each case ────────────────────────────────────────────────────────────
for (mat, thick), phits_path in cases:
    phits_flux, _ = load_phits(str(phits_path))
    if phits_flux is None:
        print(f'  skip {phits_path.name}')
        continue

    pred = cnn_flux(mat, thick)
    phits_dose = dose_midpoint(phits_flux, R_ALL, excl_bin0=True)
    cnn_dose   = dose_midpoint(pred,       R_ALL, excl_bin0=True)
    pct_err    = 100 * (cnn_dose - phits_dose) / phits_dose
    color      = MAT_COLOR.get(mat, '#333333')

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.semilogy(E_ALL, phits_flux, color='black', lw=2.0, label='PHITS', zorder=5)
    ax.semilogy(E_ALL, pred,       color=color,   lw=1.8, ls='--',
                label='CNN model', zorder=4)

    ax.set_xlim(0, 250)
    ax.set_xlabel('Neutron energy (MeV)', fontsize=11)
    ax.set_ylabel('Flux  [n / cm² / source n]', fontsize=11)
    ax.set_title(f'{mat}  —  {thick} cm', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.22)

    sign = '+' if pct_err >= 0 else ''
    fig.text(0.5, -0.02,
             f'Dose error:  {sign}{pct_err:.1f}%   '
             f'(CNN {cnn_dose:.3e}  vs  PHITS {phits_dose:.3e}  mrem/hr)',
             ha='center', fontsize=9, color='#333333')

    plt.tight_layout()
    fname = OUT / f'{mat}_{thick}cm.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  {fname.name}   dose err {sign}{pct_err:.1f}%')

print(f'\nSaved {len(cases)} PNGs → {OUT}')
