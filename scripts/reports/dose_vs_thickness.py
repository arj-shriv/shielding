"""
Dose rate vs shielding thickness for each material.
CNN model: MAE bins=50-240 (20260714)
CNN evaluated at: 10, 20, 40, 60, 80, 100, 120 cm
PHITS reference: plotted in red at available thicknesses

Usage:
    python scripts/reports/dose_vs_thickness.py [--out dose_vs_thickness.pdf] [--output-dir reports/pdf/]
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

# (label, folder, scale, colour, linestyle)
CNN_MODELS = [
    ('CNN MAE bins=50-240',    str(MODELS_DIR / 'k25-k11-k3-mae-bins50-240' / 'v1'), 0.001, 'steelblue',  'o-'),
    ('CNN MAE scale=0.1',      str(MODELS_DIR / 'k25-k11-k3-mae-scale01'    / 'v1'), 0.1,   'darkorange', 's-'),
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

parser = argparse.ArgumentParser(description='Dose vs thickness plots')
parser.add_argument('--out', default=None, help='Output PDF filename (inside --output-dir)')
parser.add_argument('--output-dir', default=None, help='Output directory (default: reports/single_layer/pdf/)')
args = parser.parse_args()

out_dir = args.output_dir if args.output_dir else str(REPORTS_SINGLE / 'pdf')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, args.out if args.out else 'dose_vs_thickness.pdf')

# ── ICRP dose conversion ────────────────────────────────────────────────────
_R_ALL = _build_r_all(DOSE_TABLE)

def dose_excl_bin0(flux):
    return dose_midpoint(flux, _R_ALL, excl_bin0=True)

def cnn_dose(model, mat, thick, scale=0.001):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    pred = 10**model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_excl_bin0(pred)

# ── Load models & spectrum ──────────────────────────────────────────────────
A_SPEC = load_spectrum(TRACKNET10_SPECTRUM)

loaded_models = []
for lbl, folder, scale, color, ls in CNN_MODELS:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    print(f'Loading: {lbl}')
    loaded_models.append((lbl, load_model(pkl_path), scale, color, ls))

# ── Build PDF ───────────────────────────────────────────────────────────────
with PdfPages(out_path) as pdf:
    for mat in ['Concrete', 'Steel', 'BPE']:
        phits_x, phits_y = [], []
        for thick, path in PHITS_FILES.get(mat, []):
            if os.path.exists(path):
                flux = load_phits_flux(path)
                if flux is not None:
                    phits_x.append(thick)
                    phits_y.append(dose_excl_bin0(flux))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f0f0')

        for lbl, mdl, scale, color, ls in loaded_models:
            doses = [cnn_dose(mdl, mat, t, scale) for t in THICKNESSES]
            ax.plot(THICKNESSES, doses, ls, color=color, linewidth=2.0,
                    markersize=7, label=lbl)

        if phits_x:
            ax.plot(phits_x, phits_y, 'o', color='crimson', markersize=10,
                    linewidth=0, zorder=5, label='PHITS (adj v1)')
            for px, py in zip(phits_x, phits_y):
                ax.annotate(f'{py:.2e}', (px, py),
                            textcoords='offset points', xytext=(6, 4),
                            fontsize=8, color='crimson')

        ax.set_xlabel('Thickness (cm)', fontsize=12)
        ax.set_ylabel('Dose rate  (mrem/hr)', fontsize=11)
        ax.set_title(f'{mat} — Dose Rate vs Thickness\n'
                     f'CNN evaluated at {THICKNESSES} cm  |  PHITS points in red',
                     fontsize=11, fontweight='bold', color='#1a252f')
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_xticks(THICKNESSES)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Page done: {mat}')

print(f'\nSaved → {out_path}')
