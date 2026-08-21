"""
One page per CNN model. Each page has 3 lines (one per material: Concrete, Steel, BPE)
showing dose rate vs thickness (20-180 cm, 10 points).
PHITS reference points overlaid per material.

Usage:
    python scripts/reports/dose_by_model.py [--out dose_by_model.pdf] [--output-dir reports/pdf/]
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from shielding_ml.data.loaders import load_phits_flux, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, MODEL_PKL_NAME
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    PHITS_HE, DOSE_TABLE, TRACKNET10_SPECTRUM, MODELS_DIR, REPORTS_SINGLE,
)

THICKNESSES = list(np.linspace(20, 180, 10).astype(int))

# (label, folder, scale)
MODELS = [
    ('TrackNet10',                  str(MODELS_DIR / 'tracknet10-original'              / 'v1'), 0.001),
    ('Piecewise6',                  str(MODELS_DIR / 'piecewise6'                       / 'v1'), 0.001),
    ('Piecewise6 off=1.1',          str(MODELS_DIR / 'piecewise6-offset-1.1'           / 'v1'), 0.001),
    ('Piecewise6 off=1.2',          str(MODELS_DIR / 'piecewise6-offset-1.2'           / 'v1'), 0.001),
    ('Piecewise6 off=1.3',          str(MODELS_DIR / 'piecewise6-offset-1.3'           / 'v1'), 0.001),
    ('Piecewise6 off=1.4',          str(MODELS_DIR / 'piecewise6-offset-1.4'           / 'v1'), 0.001),
    ('k11,k3  d256/128  MSE',       str(MODELS_DIR / 'k11-k3-d256-d128-mse'            / 'v1'), 0.001),
    ('k25,k11,k3  MSE',             str(MODELS_DIR / 'k25-k11-k3-mse'                  / 'v1'), 0.001),
    ('k25,k11,k3  MAE',             str(MODELS_DIR / 'k25-k11-k3-mae'                  / 'v1'), 0.001),
    ('MAE  scale=0.0005',           str(MODELS_DIR / 'k25-k11-k3-mae-scale0005'        / 'v1'), 0.0005),
    ('MAE  scale=0.1',              str(MODELS_DIR / 'k25-k11-k3-mae-scale01'          / 'v1'), 0.1),
    ('MAE  bins=240',               str(MODELS_DIR / 'k25-k11-k3-mae-bins240'          / 'v1'), 0.001),
    ('MAE  bins=50-240',            str(MODELS_DIR / 'k25-k11-k3-mae-bins50-240'       / 'v1'), 0.001),
    ('k51,k11,k3  s1  bins50-240',  str(MODELS_DIR / 'k51-k11-k3-mae-s1-bins50-240'   / 'v1'), 0.001),
    ('k51,k11,k3  s2  bins50-240',  str(MODELS_DIR / 'k51-k11-k3-mae-s2-bins50-240'   / 'v1'), 0.001),
]

PHITS_FILES = {
    'Concrete': [
        (25,  str(PHITS_HE / 'Concrete_25cm_1.out')),
        (37,  str(PHITS_HE / 'Concrete_37cm_1')),
        (75,  str(PHITS_HE / 'Concrete_75cm_1.out')),
        (117, str(PHITS_HE / 'Concrete_117cm_1.out')),
    ],
    'Steel': [
        (27,  str(PHITS_HE / 'Steel_27cm_1.out')),
        (55,  str(PHITS_HE / 'Steel_55cm_1.out')),
        (73,  str(PHITS_HE / 'Steel_73cm_1.out')),
    ],
    'BPE': [
        (25,  str(PHITS_HE / 'BPE_25cm.out')),
        (45,  str(PHITS_HE / 'BPE_45cm.out')),
        (63,  str(PHITS_HE / 'BPE_63cm.out')),
    ],
}

MAT_STYLE = {
    'Concrete': ('steelblue',   'o'),
    'Steel':    ('darkorange',  's'),
    'BPE':      ('seagreen',    '^'),
}

parser = argparse.ArgumentParser(description='Dose by model plots')
parser.add_argument('--out', default=None, help='Output PDF filename (inside --output-dir)')
parser.add_argument('--output-dir', default=None, help='Output directory (default: reports/single_layer/pdf/)')
args = parser.parse_args()

out_dir = args.output_dir if args.output_dir else str(REPORTS_SINGLE / 'pdf')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, args.out if args.out else 'dose_by_model.pdf')

# ── ICRP dose conversion ─────────────────────────────────────────────────────
_R_ALL = _build_r_all(DOSE_TABLE)

def dose_excl_bin0(flux):
    return dose_midpoint(flux, _R_ALL, excl_bin0=True)

def cnn_dose(model, mat, thick, scale):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    pred = 10**model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_excl_bin0(pred)

# ── Pre-compute PHITS reference doses ────────────────────────────────────────
A_SPEC = load_spectrum(TRACKNET10_SPECTRUM)

phits_ref = {}
for mat, entries in PHITS_FILES.items():
    xs, ys = [], []
    for thick, path in entries:
        if os.path.exists(path):
            flux = load_phits_flux(path)
            if flux is not None:
                xs.append(thick)
                ys.append(dose_excl_bin0(flux))
    phits_ref[mat] = (xs, ys)

# ── Build PDF ─────────────────────────────────────────────────────────────────
with PdfPages(out_path) as pdf:
    for model_lbl, folder, scale in MODELS:
        pkl_path = os.path.join(folder, MODEL_PKL_NAME)
        if not os.path.exists(pkl_path):
            print(f'  Skipping {model_lbl} — pkl not found')
            continue

        print(f'Loading: {model_lbl}')
        model = load_model(pkl_path)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f0f0')

        for mat in ['Concrete', 'Steel', 'BPE']:
            color, marker = MAT_STYLE[mat]

            doses = [cnn_dose(model, mat, t, scale) for t in THICKNESSES]
            ax.plot(THICKNESSES, doses, f'{marker}-', color=color, linewidth=2.0,
                    markersize=7, label=f'{mat} (CNN)')

            px, py = phits_ref[mat]
            if px:
                ax.plot(px, py, marker, color=color, markersize=11,
                        markerfacecolor='white', markeredgewidth=2,
                        linewidth=0, zorder=5, label=f'{mat} (PHITS)')

        ax.set_xlabel('Thickness (cm)', fontsize=12)
        ax.set_ylabel('Dose rate  (mrem/hr)', fontsize=11)
        ax.set_title(f'{model_lbl}\nDose Rate vs Thickness — all materials',
                     fontsize=11, fontweight='bold', color='#1a252f')
        ax.set_xticks(THICKNESSES)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Page done: {model_lbl}')

print(f'\nSaved → {out_path}')
