"""
Generate result_content/comparison.pdf

One page per case (material + thickness).
Each page:
  - Flux panel  : PHITS vs TrackNet10 vs MAE scale=0.1  (semilogy)
  - Ratio panel : pred / PHITS for both models
  - Dose table  : PHITS dose | TrackNet10 dose + % err | scale=0.1 dose + % err

Usage:
    python scripts/reports/make_result_content.py [--output-dir reports/pdf/]
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL, MODEL_PKL_NAME
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model, build_input
from shielding_ml.pipelines.paths import (
    PHITS_HE, DOSE_TABLE, TRACKNET10_SPECTRUM, MODELS_DIR, REPORTS_SINGLE,
)

parser = argparse.ArgumentParser(description='Generate comparison PDF per shielding case')
parser.add_argument('--output-dir', default=None,
                    help='Output directory (default: reports/single_layer/pdf/)')
args = parser.parse_args()

OUT_DIR = args.output_dir if args.output_dir else str(REPORTS_SINGLE / 'pdf')
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ('Original CNN', str(MODELS_DIR / 'tracknet10-original'  / 'v1'), 0.001, '#e8734a'),
    ('New CNN',      str(MODELS_DIR / 'k25-k11-k3-mae-scale01' / 'v1'), 0.1, '#4a90d9'),
]

CASES = [
    ('Concrete  25 cm', 'Concrete', 25,  str(PHITS_HE / 'Concrete_25cm_1.out')),
    ('Concrete  75 cm', 'Concrete', 75,  str(PHITS_HE / 'Concrete_75cm_1.out')),
    ('Concrete 117 cm', 'Concrete', 117, str(PHITS_HE / 'Concrete_117cm_1.out')),
    ('Steel  27 cm',    'Steel',    27,  str(PHITS_HE / 'Steel_27cm_1.out')),
    ('Steel  55 cm',    'Steel',    55,  str(PHITS_HE / 'Steel_55cm_1.out')),
    ('Steel  73 cm',    'Steel',    73,  str(PHITS_HE / 'Steel_73cm_1.out')),
    ('BPE  25 cm',      'BPE',      25,  str(PHITS_HE / 'BPE_25cm.out')),
    ('BPE  45 cm',      'BPE',      45,  str(PHITS_HE / 'BPE_45cm.out')),
    ('BPE  63 cm',      'BPE',      63,  str(PHITS_HE / 'BPE_63cm.out')),
    ('Concrete  37 cm', 'Concrete', 37,  str(PHITS_HE / 'Concrete_37cm_1')),
]

# ── ICRP dose conversion ─────────────────────────────────────────────────────
_R_ALL = _build_r_all(DOSE_TABLE)

def dose_excl_bin0(flux):
    return dose_midpoint(flux, _R_ALL, excl_bin0=True)

def _err_color(pct):
    v = abs(pct)
    if v < 20:  return (0.55, 0.86, 0.60, 0.75)
    if v < 30:  return (1.00, 0.93, 0.45, 0.75)
    return (0.95, 0.45, 0.45, 0.75)

def cnn_predict(model, mat, thick, scale):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    return 10**model.predict(X, verbose=0).flatten() * FLUX_SCALE

# ── Load models ───────────────────────────────────────────────────────────────
A_SPEC = load_spectrum(TRACKNET10_SPECTRUM)

loaded = []
for lbl, folder, scale, color in MODELS:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    if not os.path.exists(pkl_path):
        print(f'  Skipping {lbl} — pkl not found')
        continue
    print(f'Loading: {lbl}')
    loaded.append((lbl, load_model(pkl_path), scale, color))

# ── Build PDF ─────────────────────────────────────────────────────────────────
out_pdf = os.path.join(OUT_DIR, 'comparison.pdf')
print(f'\nWriting → {out_pdf}')

with PdfPages(out_pdf) as pdf:
    for case_lbl, mat, thick, phits_path in CASES:
        if not os.path.exists(phits_path):
            print(f'  Skipping — not found: {phits_path}')
            continue

        phits_flux, phits_rerr = load_phits(phits_path)
        if phits_flux is None:
            print(f'  Parse failed: {phits_path}')
            continue

        phits_dose = dose_excl_bin0(phits_flux)
        phits_sig  = phits_flux * np.nan_to_num(phits_rerr)

        model_preds = []
        for lbl, model, scale, color in loaded:
            pred = cnn_predict(model, mat, thick, scale)
            d    = dose_excl_bin0(pred)
            pct  = 100 * (d - phits_dose) / phits_dose
            model_preds.append((lbl, pred, d, pct, color))

        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(3, 1, height_ratios=[2.6, 1.2, 0.9],
                               hspace=0.45, figure=fig)
        ax_flux = fig.add_subplot(gs[0])
        ax_rat  = fig.add_subplot(gs[1])
        ax_tbl  = fig.add_subplot(gs[2])

        ax_flux.semilogy(E_ALL[1:], phits_flux[1:], color='black',
                         linewidth=1.4, alpha=0.45, label='PHITS', zorder=5)
        ax_flux.fill_between(E_ALL[1:],
                             np.clip(phits_flux[1:] - phits_sig[1:], 1e-30, None),
                             phits_flux[1:] + phits_sig[1:],
                             color='black', alpha=0.06, zorder=4)
        for lbl, pred, d, pct, color in model_preds:
            ax_flux.semilogy(E_ALL[1:], pred[1:], color=color,
                             linewidth=1.4, linestyle='--', label=lbl, zorder=3)

        ax_flux.set_xlim(1, 250)
        ax_flux.set_ylim(bottom=1e-7)
        ax_flux.set_xlabel('Energy (MeV)', fontsize=11)
        ax_flux.set_ylabel('Neutron flux  [n/cm²/MeV/source]', fontsize=10)
        ax_flux.legend(fontsize=9, framealpha=0.85)
        ax_flux.grid(True, which='both', alpha=0.2)
        ax_flux.set_facecolor('#f0f2f5')

        ax_rat.axhline(1, color='black', linewidth=1.0, linestyle='--', zorder=5)
        ax_rat.axhspan(0.8, 1.2, alpha=0.07, color='green')
        all_ratios = []
        for lbl, pred, d, pct, color in model_preds:
            mask  = (phits_flux[1:] > 0) & np.isfinite(pred[1:])
            ratio = np.where(mask, pred[1:] / phits_flux[1:], np.nan)
            ax_rat.plot(E_ALL[1:], ratio, color=color, linewidth=1.0, label=lbl)
            valid = ratio[np.isfinite(ratio)]
            if len(valid): all_ratios.append(valid)

        if all_ratios:
            combined = np.concatenate(all_ratios)
            rlo = max(0, np.nanpercentile(combined, 2)  * 0.9)
            rhi = min(10, np.nanpercentile(combined, 98) * 1.1)
            ax_rat.set_ylim(rlo, rhi)

        ax_rat.set_xlim(1, 250)
        ax_rat.set_xlabel('Energy (MeV)', fontsize=11)
        ax_rat.set_ylabel('CNN / PHITS', fontsize=10)
        ax_rat.legend(fontsize=8, loc='upper right')
        ax_rat.grid(True, alpha=0.25)
        ax_rat.set_facecolor('#f0f2f5')

        ax_tbl.axis('off')
        col_labels = ['', 'Dose (1–250 MeV)', '% err vs PHITS']
        rows = [['PHITS', f'{phits_dose:.4e}', '—']]
        cell_colors = [['#2c3e50', '#d0e8ff', '#d0e8ff']]
        for lbl, pred, d, pct, color in model_preds:
            rows.append([lbl, f'{d:.4e}', f'{pct:+.1f}%'])
            cell_colors.append(['#f0f0f0', 'white', _err_color(pct)])

        tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                           cellColours=cell_colors,
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)
        for j in range(3):
            tbl[0, j].set_facecolor('#1a252f')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        tbl[1, 0].set_text_props(color='white', fontweight='bold')

        fig.suptitle(f'{case_lbl}  —  {mat}  {thick} cm',
                     fontsize=13, fontweight='bold', color='#1a252f', y=1.01)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Done: {case_lbl}')

print(f'\nSaved → {out_pdf}')
