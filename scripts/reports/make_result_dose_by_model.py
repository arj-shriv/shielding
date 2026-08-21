"""
result_content/dose_by_model.pdf
One page per material (Concrete, Steel, BPE):
  - CNN dose vs thickness curve for Original CNN (TrackNet10) and New CNN (scale=0.1)
  - PHITS reference points overlaid per material

Usage:
    python scripts/reports/make_result_dose_by_model.py [--output-dir reports/pdf/]
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

# (label, folder, scale, color, linestyle)
MODELS = [
    ('Original CNN', str(MODELS_DIR / 'tracknet10-original'    / 'v1'), 0.001, '#e8734a', 'o-'),
    ('New CNN',      str(MODELS_DIR / 'k25-k11-k3-mae-scale01' / 'v1'), 0.1,   '#4a90d9', 's-'),
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

parser = argparse.ArgumentParser(description='Result dose by model plots')
parser.add_argument('--output-dir', default=None, help='Output directory (default: reports/single_layer/pdf/)')
args = parser.parse_args()

OUT_DIR = args.output_dir if args.output_dir else str(REPORTS_SINGLE / 'pdf')
os.makedirs(OUT_DIR, exist_ok=True)

# ── ICRP dose conversion ──────────────────────────────────────────────────────
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

# ── Load ─────────────────────────────────────────────────────────────────────
A_SPEC = load_spectrum(TRACKNET10_SPECTRUM)

loaded = []
for lbl, folder, scale, color, ls in MODELS:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    if not os.path.exists(pkl_path):
        print(f'  Skipping {lbl} — pkl not found')
        continue
    print(f'Loading: {lbl}')
    loaded.append((lbl, load_model(pkl_path), scale, color, ls))

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
out_pdf = os.path.join(OUT_DIR, 'dose_by_model.pdf')
print(f'\nWriting → {out_pdf}')

mat_marker = {'Concrete': 'o', 'Steel': 's', 'BPE': '^'}

with PdfPages(out_pdf) as pdf:
    for mat in ['Concrete', 'Steel', 'BPE']:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f0f2f5')

        for lbl, mdl, scale, color, ls in loaded:
            doses = [cnn_dose(mdl, mat, t, scale) for t in THICKNESSES]
            ax.plot(THICKNESSES, doses, ls, color=color, linewidth=2.2,
                    markersize=7, label=lbl)

        px, py = phits_ref.get(mat, ([], []))
        if px:
            ax.plot(px, py, mat_marker[mat], color='black', markersize=11,
                    markerfacecolor='white', markeredgewidth=2.0,
                    linewidth=0, zorder=5, label='PHITS')
            for x, y in zip(px, py):
                ax.annotate(f'{y:.2e}', (x, y),
                            textcoords='offset points', xytext=(6, 4),
                            fontsize=8, color='#333333')

        ax.set_xlabel('Thickness (cm)', fontsize=12)
        ax.set_ylabel('Effective dose  (pSv*cm2)', fontsize=11)
        ax.set_title(f'{mat}  —  Dose vs Thickness\nOriginal CNN vs New CNN with PHITS reference',
                     fontsize=11, fontweight='bold', color='#1a252f')
        ax.set_xticks(THICKNESSES)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.25)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Page done: {mat}')

print(f'\nSaved → {out_pdf}')
