"""
% dose error: old TrackNet10 model vs new k100s2 model — single-layer shielding.

For each PHITS reference point (material, thickness), computes:
    % error = (CNN_dose - PHITS_dose) / PHITS_dose × 100

Output: inference_results/dose_error_comparison/dose_pct_error.pdf
"""

from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.pipelines.paths import REPO_ROOT, TRACKNET10_SPECTRUM, DOSE_TABLE
from shielding_ml.data.constants import FLUX_SCALE
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint

# ── paths ─────────────────────────────────────────────────────────────────────
_PY_CODES = Path('/Users/arjun/Python_Codes')
_OLD_PKL  = (_PY_CODES / '20260706_TrackNet10' /
             '1D_CNN_GaussianBroadenedImpulse_MAE_ShieldingThickness_150MeV_'
             '10000samples_TrackNet10Weights_LogValues_kernel_3_'
             'CombinedMaterials_epoch20.pkl')
_NEW_PKL  = (REPO_ROOT / 'models'
             / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
             / 'v1' / 'model.pkl')

_PHITS_DIR = _PY_CODES / 'Actual_Output' / 'Adjusted_Output'

OUT_DIR = REPO_ROOT / 'inference_results' / 'dose_error_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PHITS reference points ─────────────────────────────────────────────────────
PHITS_FILES = {
    'Concrete': [
        (25,  'Concrete_25cm_1.out'),
        (37,  'Concrete_37cm_1'),
        (75,  'Concrete_75cm_1.out'),
        (117, 'Concrete_117cm_1.out'),
    ],
    'Steel': [
        (27, 'Steel_27cm_1.out'),
        (55, 'Steel_55cm_1.out'),
        (73, 'Steel_73cm_1.out'),
    ],
    'BPE': [
        (25, 'BPE_25cm.out'),
        (45, 'BPE_45cm.out'),
        (63, 'BPE_63cm.out'),
    ],
}

DENSITY    = {'Concrete': 2.3, 'Steel': 7.86, 'BPE': 1.04}
OLD_SCALE  = 0.001
NEW_SCALE  = 0.001

# ── load ──────────────────────────────────────────────────────────────────────
with open(_OLD_PKL, 'rb') as f: old_model = pickle.load(f)
with open(_NEW_PKL, 'rb') as f: new_model = pickle.load(f)
A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
R_ALL  = _build_r_all(DOSE_TABLE)
print('Models loaded.')

# ── helpers ───────────────────────────────────────────────────────────────────
def cnn_dose(model, scale, mat, thick):
    dens = DENSITY[mat]
    B  = np.ones(250) * thick * scale
    B1 = np.ones(250) * dens  * scale
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    flux = 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_midpoint(flux, R_ALL, excl_bin0=True)

def phits_dose(rel_path):
    flux, _ = load_phits(str(_PHITS_DIR / rel_path))
    return dose_midpoint(flux, R_ALL, excl_bin0=True) if flux is not None else None

# ── compute errors ─────────────────────────────────────────────────────────────
results = {}   # {mat: {'thick': [], 'phits': [], 'old_pct': [], 'new_pct': []}}
for mat, entries in PHITS_FILES.items():
    thicks, phits_doses, old_pcts, new_pcts = [], [], [], []
    for thick, fname in entries:
        pd = phits_dose(fname)
        if pd is None:
            print(f'  Skipping {mat} {thick}cm — PHITS load failed')
            continue
        od = cnn_dose(old_model, OLD_SCALE, mat, thick)
        nd = cnn_dose(new_model, NEW_SCALE, mat, thick)
        thicks.append(thick)
        phits_doses.append(pd)
        old_pcts.append(100 * (od - pd) / pd)
        new_pcts.append(100 * (nd - pd) / pd)
        print(f'  {mat} {thick}cm  PHITS={pd:.3e}  '
              f'old={od:.3e} ({old_pcts[-1]:+.1f}%)  '
              f'new={nd:.3e} ({new_pcts[-1]:+.1f}%)')
    results[mat] = dict(thick=thicks, phits=phits_doses,
                        old_pct=old_pcts, new_pct=new_pcts)

# ── plot ───────────────────────────────────────────────────────────────────────
MAT_COLOR  = {'Concrete': '#4e79a7', 'Steel': '#e15759', 'BPE': '#59a14f'}
OLD_STYLE  = dict(ls='--', marker='o', lw=1.8, ms=8, alpha=0.8)
NEW_STYLE  = dict(ls='-',  marker='s', lw=2.2, ms=8, alpha=0.95)

OUT_PDF = OUT_DIR / 'dose_pct_error.pdf'
with PdfPages(OUT_PDF) as pdf:

    # ── Page 1: one subplot per material ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Single-layer dose error: old TrackNet10 model vs new k100s2',
                 fontsize=13, fontweight='bold')

    for ax, mat in zip(axes, ['Concrete', 'Steel', 'BPE']):
        r     = results[mat]
        color = MAT_COLOR[mat]

        ax.plot(r['thick'], r['old_pct'], color='#b0b0b0', label='Old (TrackNet10)',
                **OLD_STYLE)
        ax.plot(r['thick'], r['new_pct'], color=color,    label='New (k100s2)',
                **NEW_STYLE)

        # Annotate each point with its value
        for x, y in zip(r['thick'], r['old_pct']):
            ax.annotate(f'{y:+.0f}%', (x, y), textcoords='offset points',
                        xytext=(-6, 8), fontsize=7.5, color='#888888')
        for x, y in zip(r['thick'], r['new_pct']):
            ax.annotate(f'{y:+.0f}%', (x, y), textcoords='offset points',
                        xytext=(-6, -14), fontsize=7.5, color=color)

        ax.axhline(0, color='k', lw=1, ls='-', alpha=0.4)
        ax.axhspan(-20, 20, color='green', alpha=0.06, label='±20% band')

        ax.set_title(f'{mat}  (ρ = {DENSITY[mat]} g/cm³)', fontsize=10,
                     fontweight='bold')
        ax.set_xlabel('Thickness (cm)', fontsize=9)
        ax.set_ylabel('Dose error (%)', fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_xticks(r['thick'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # ── Page 2: all materials on one plot with shared y-axis ──────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle('Dose % error — all materials',
                 fontsize=12, fontweight='bold')

    x_pos = 0
    tick_pos, tick_lbl = [], []
    group_gap = 1.5

    for mat in ['Concrete', 'Steel', 'BPE']:
        r     = results[mat]
        color = MAT_COLOR[mat]
        n     = len(r['thick'])
        xs    = np.arange(x_pos, x_pos + n)

        ax.plot(xs, r['old_pct'], color='#aaaaaa', **OLD_STYLE,
                label='Old (TrackNet10)' if mat == 'Concrete' else '_')
        ax.plot(xs, r['new_pct'], color=color, **NEW_STYLE,
                label=f'New k100s2 — {mat}')

        for xi, (t, op, np_) in enumerate(zip(r['thick'], r['old_pct'], r['new_pct'])):
            tick_pos.append(x_pos + xi)
            tick_lbl.append(f'{t}')

        # Material label under group
        ax.annotate(mat, xy=((x_pos + x_pos + n - 1) / 2, ax.get_ylim()[0] if False else -1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold', color=color,
                    annotation_clip=False)

        x_pos += n + group_gap

    ax.axhline(0, color='k', lw=1, ls='-', alpha=0.4)
    ax.axhspan(-20, 20, color='green', alpha=0.06, label='±20% band')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=8)
    ax.set_xlabel('Thickness (cm)', fontsize=10)
    ax.set_ylabel('Dose error (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f'\nSaved → {OUT_PDF}')
