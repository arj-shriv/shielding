"""
Multi-page PDF: one page per material+thickness case.
Columns: Model | PHITS HE +-sigma% | PHITS LE | CNN HE | % err HE | CNN Total (LE+HE) | % err Total | Flux MAPE

PHITS HE : 250 bins 0-250 MeV (rectangle rule, 1 MeV bins, full ICRP table)
PHITS LE : 100 log-spaced bins 1 keV-1 MeV from LEOut files
CNN Total: PHITS LE + CNN HE (excl. bin-0 overlap) - LE range not predicted by CNN

Usage:
    python scripts/reports/make_dose_pdf.py [--out dose_comparison.pdf] [--output-dir reports/pdf/]
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from shielding_ml.data.loaders import load_phits, load_le_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL, MODEL_PKL_NAME
from shielding_ml.metrics.dose import (
    _build_r_all, dose_midpoint, integrate_le_dose,
)
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    PHITS_HE, PHITS_LE, DOSE_TABLE, TRACKNET10_SPECTRUM, MODELS_DIR, REPORTS_SINGLE,
)

# (row_label, folder, scale)
MODELS = [
    ('TrackNet10',              str(MODELS_DIR / 'tracknet10-original'              / 'v1'), 0.001),
    ('Piecewise6',              str(MODELS_DIR / 'piecewise6'                       / 'v1'), 0.001),
    ('Piecewise6 off=1.1',      str(MODELS_DIR / 'piecewise6-offset-1.1'           / 'v1'), 0.001),
    ('Piecewise6 off=1.2',      str(MODELS_DIR / 'piecewise6-offset-1.2'           / 'v1'), 0.001),
    ('Piecewise6 off=1.3',      str(MODELS_DIR / 'piecewise6-offset-1.3'           / 'v1'), 0.001),
    ('Piecewise6 off=1.4',      str(MODELS_DIR / 'piecewise6-offset-1.4'           / 'v1'), 0.001),
    ('k11,k3  d256/128  MSE',   str(MODELS_DIR / 'k11-k3-d256-d128-mse'            / 'v1'), 0.001),
    ('k25,k11,k3  MSE',         str(MODELS_DIR / 'k25-k11-k3-mse'                  / 'v1'), 0.001),
    ('k25,k11,k3  MAE',         str(MODELS_DIR / 'k25-k11-k3-mae'                  / 'v1'), 0.001),
    ('MAE  scale=0.0005',       str(MODELS_DIR / 'k25-k11-k3-mae-scale0005'        / 'v1'), 0.0005),
    ('MAE  scale=0.1',          str(MODELS_DIR / 'k25-k11-k3-mae-scale01'          / 'v1'), 0.1),
    ('MAE  bins=240',           str(MODELS_DIR / 'k25-k11-k3-mae-bins240'          / 'v1'), 0.001),
    ('MAE  bins=50-240',        str(MODELS_DIR / 'k25-k11-k3-mae-bins50-240'       / 'v1'), 0.001),
    ('k51,k11,k3  s1  bins50-240', str(MODELS_DIR / 'k51-k11-k3-mae-s1-bins50-240' / 'v1'), 0.001),
    ('k51,k11,k3  s2  bins50-240', str(MODELS_DIR / 'k51-k11-k3-mae-s2-bins50-240' / 'v1'), 0.001),
]

# (page title, material, thickness, he_phits_path or None, le_phits_path or None)
CASES = [
    ('Concrete  25 cm (adj v1)', 'Concrete', 25,
     str(PHITS_HE / 'Concrete_25cm_1.out'),
     str(PHITS_LE / 'Concrete_25cm_1_LE.out')),
    ('Concrete  75 cm (adj v1)', 'Concrete', 75,
     str(PHITS_HE / 'Concrete_75cm_1.out'),
     str(PHITS_LE / 'Concrete_75cm_1_LE.out')),
    ('Concrete 117 cm (adj v1)', 'Concrete', 117,
     str(PHITS_HE / 'Concrete_117cm_1.out'),
     str(PHITS_LE / 'Concrete_117cm_1_LE.out')),
    ('Steel  27 cm',             'Steel',    27,
     str(PHITS_HE / 'Steel_27cm_1.out'),
     str(PHITS_LE / 'Steel_27cm_1_LE.out')),
    ('Steel  55 cm (adj v1)',    'Steel',    55,
     str(PHITS_HE / 'Steel_55cm_1.out'),
     str(PHITS_LE / 'Steel_55cm_1_LE.out')),
    ('Steel  73 cm',             'Steel',    73,
     str(PHITS_HE / 'Steel_73cm_1.out'),
     str(PHITS_LE / 'Steel_73cm_1_LE.out')),
    ('BPE  25 cm',               'BPE',      25,
     str(PHITS_HE / 'BPE_25cm.out'),
     None),
    ('BPE  45 cm',               'BPE',      45,
     str(PHITS_HE / 'BPE_45cm.out'),
     None),
    ('BPE  63 cm',               'BPE',      63,
     str(PHITS_HE / 'BPE_63cm.out'),
     None),
    ('Concrete  37 cm',          'Concrete', 37,
     str(PHITS_HE / 'Concrete_37cm_1'),
     None),
]

parser = argparse.ArgumentParser(description='Multi-page dose comparison PDF')
parser.add_argument('--out', default=None, help='Output PDF filename (inside --output-dir)')
parser.add_argument('--output-dir', default=None, help='Output directory (default: reports/single_layer/pdf/)')
args = parser.parse_args()

out_dir = args.output_dir if args.output_dir else str(REPORTS_SINGLE / 'pdf')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, args.out if args.out else 'dose_comparison.pdf')

# ── Dose integration setup ────────────────────────────────────────────────────
_R_ALL = _build_r_all(DOSE_TABLE)
_E_tbl, _R_tbl = np.loadtxt(DOSE_TABLE, unpack=True)
_f1 = interp1d(_E_tbl, _R_tbl)

def integrate_he_dose(flux_all):
    """HE dose: all 250 bins 0-250 MeV (rectangle rule, deltaE = 1 MeV)."""
    return float(np.sum(flux_all * _R_ALL))

def integrate_he_dose_excl_bin0(flux_all):
    """HE dose excluding bin-0 (0-1 MeV) to avoid overlap with LE range."""
    return float(np.sum(flux_all[1:] * _R_ALL[1:]))

def _integrate_le_dose(e_lo, e_hi, flux_arr):
    """LE dose: variable deltaE = e_hi - e_lo (log-spaced bins), rectangle rule."""
    e_mid = (e_lo + e_hi) / 2.0
    dE    = e_hi - e_lo
    R     = _f1(np.clip(e_mid, _E_tbl[0], _E_tbl[-1]))
    return float(np.sum(flux_arr * R * dE))

def compute_mape(pred_flux, actual_flux):
    if actual_flux is None:
        return float('nan')
    mask = (actual_flux > 0) & (np.arange(250) < 249)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred_flux[mask] - actual_flux[mask]) / actual_flux[mask]) * 100)

def cnn_predict(model, mat, thick, scale):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([A, B, B1]).T.reshape(1, 250, 3)
    return 10**model.predict(X, verbose=0).flatten() * FLUX_SCALE

# ── Formatting ────────────────────────────────────────────────────────────────
def fe(v):
    if v is None or np.isnan(v): return '—'
    s = f'{v:.3e}'
    m, e = s.split('e')
    return f'{m}x10^{int(e)}'

def dose_err_color(pct, alpha=0.6):
    if pct is None or np.isnan(pct): return (0.88, 0.88, 0.88, 0.4)
    v = abs(pct)
    if v < 20:   return (0.55, 0.86, 0.60, alpha)
    elif v < 30: return (1.00, 0.93, 0.50, alpha)
    else:        return (0.95, 0.50, 0.45, alpha)

def mape_color(pct, alpha=0.6):
    if pct is None or np.isnan(pct): return (0.88, 0.88, 0.88, 0.4)
    v = abs(pct)
    if v < 20:   return (0.55, 0.86, 0.60, alpha)
    elif v < 50: return (1.00, 0.93, 0.50, alpha)
    else:        return (0.95, 0.50, 0.45, alpha)

def fmt_pct(pct):
    if pct is None or np.isnan(pct): return '—'
    return f'{pct:+.1f}%'

def summary_color(v):
    if np.isnan(v): return (0.85, 0.85, 0.85, 1.0)
    if abs(v) < 20:  return (0.55, 0.86, 0.60, 1.0)
    if abs(v) < 30:  return (1.00, 0.93, 0.45, 1.0)
    return (0.95, 0.45, 0.45, 1.0)

# ── Load all models ───────────────────────────────────────────────────────────
A = load_spectrum(TRACKNET10_SPECTRUM)

loaded_models = []
for label, folder, scale in MODELS:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    if os.path.exists(pkl_path):
        print(f'Loading {label} ...')
        loaded_models.append((label, load_model(pkl_path), scale))
    else:
        print(f'SKIP (not found): {folder}')
        loaded_models.append((label, None, scale))

# ── Column layout ─────────────────────────────────────────────────────────────
COL_LABELS = [
    'Model',
    'PHITS\n(1-250 MeV)\n(pSv*cm2) +/-sigma%',
    'CNN\n(1-250 MeV)\n(pSv*cm2)',
    '% err\n(1-250 MeV)',
    'PHITS LE\n(1keV-1MeV)\n(pSv*cm2)',
    'Flux MAPE\n|y_hat-y|/y',
]
N_COLS   = len(COL_LABELS)
ROW_EVEN = (0.97, 0.97, 0.97, 1.0)
ROW_ODD  = (1.00, 1.00, 1.00, 1.0)
CELL_NA  = (0.90, 0.90, 0.90, 0.5)
PHITS_BG = (0.82, 0.88, 0.95, 1.0)
LE_BG    = (0.88, 0.95, 0.88, 1.0)

all_errors = {label: [] for label, _, _ in MODELS}
all_mapes  = {label: [] for label, _, _ in MODELS}

with PdfPages(out_path) as pdf:
    for case_title, mat, thick, he_path, le_path in CASES:

        # ── Load HE PHITS ───────────────────────────────────────────────────
        actual, sigma = None, None
        d_he_phits    = float('nan')
        phits_he_str  = '—'
        if he_path and os.path.exists(he_path):
            actual, sigma = load_phits(he_path)
            if actual is not None:
                d_he_phits = integrate_he_dose_excl_bin0(actual)
                have_sigma = sigma is not None and not np.all(np.isnan(sigma))
                if have_sigma:
                    d_lo = integrate_he_dose_excl_bin0(np.clip(actual - sigma, 0, None))
                    d_hi = integrate_he_dose_excl_bin0(np.clip(actual + sigma, 0, None))
                    sp   = 100 * (d_hi - d_lo) / (2 * d_he_phits)
                    phits_he_str = f'{fe(d_he_phits)}\n+/-{sp:.3f}%'
                else:
                    phits_he_str = f'{fe(d_he_phits)}\n(no sigma)'
        elif he_path:
            print(f'  HE file not found: {he_path}')

        # ── Load LE PHITS ───────────────────────────────────────────────────
        d_le_phits   = float('nan')
        phits_le_str = '—'
        if le_path and os.path.exists(le_path):
            le_elo, le_ehi, le_flux, le_sig = load_le_phits(le_path)
            if le_flux is not None:
                d_le_phits   = _integrate_le_dose(le_elo, le_ehi, le_flux)
                phits_le_str = fe(d_le_phits)
        elif le_path:
            print(f'  LE file not found: {le_path}')

        # ── Build table rows (one per model) ────────────────────────────────
        cell_text, cell_colors = [], []
        for i, (label, model, scale) in enumerate(loaded_models):
            bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
            if model is None:
                cell_text.append([label, phits_he_str, '—', '—', phits_le_str, '—'])
                cell_colors.append([bg, PHITS_BG, CELL_NA, CELL_NA, LE_BG, CELL_NA])
                continue

            pred_flux = cnn_predict(model, mat, thick, scale)
            d_cnn_he  = integrate_he_dose_excl_bin0(pred_flux)
            pct_he    = 100 * (d_cnn_he - d_he_phits) / d_he_phits if not np.isnan(d_he_phits) else float('nan')
            mape      = compute_mape(pred_flux, actual)
            if not np.isnan(pct_he):
                all_errors[label].append(abs(pct_he))
            if not np.isnan(mape):
                all_mapes[label].append(mape)

            cell_text.append([
                label,
                phits_he_str,
                fe(d_cnn_he),
                fmt_pct(pct_he),
                phits_le_str,
                f'{mape:.1f}%' if not np.isnan(mape) else '—',
            ])
            cell_colors.append([
                bg, PHITS_BG,
                bg, dose_err_color(pct_he),
                LE_BG, mape_color(mape),
            ])

        # ── Render page ──────────────────────────────────────────────────────
        n_rows = len(cell_text)
        fig, ax = plt.subplots(figsize=(19, 1.8 + n_rows * 0.80))
        ax.axis('off')
        fig.patch.set_facecolor('#f5f7fa')

        tbl = ax.table(cellText=cell_text, colLabels=COL_LABELS, cellColours=cell_colors,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 2.3)

        for j in range(N_COLS):
            tbl[0, j].set_facecolor('#2c3e50')
            tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=8.5)

        for r in range(1, n_rows + 1):
            tbl[r, 0].set_text_props(ha='left')
            tbl[r, 0]._loc = 'left'

        phits_label = f'PHITS (1-250 MeV) = {fe(d_he_phits)} pSv*cm2' if not np.isnan(d_he_phits) else 'PHITS (1-250 MeV) = —'
        le_label    = f'PHITS LE = {fe(d_le_phits)} pSv*cm2' if not np.isnan(d_le_phits) else ''
        ax.set_title(
            f'Integrated Effective Dose Rate — {case_title}\n'
            f'{phits_label}     {le_label}\n'
            '% err = (CNN - PHITS) / PHITS   (both 1-250 MeV)',
            fontsize=10, fontweight='bold', pad=14, color='#1a252f'
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Page done: {case_title}')

    # ── Summary slide ─────────────────────────────────────────────────────────
    labels    = [lbl for lbl, _, _ in MODELS]
    x         = np.arange(len(labels))
    mean_dose = [np.mean(all_errors[lbl]) if all_errors[lbl] else float('nan') for lbl in labels]
    mean_mape = [np.mean(all_mapes[lbl])  if all_mapes[lbl]  else float('nan') for lbl in labels]

    def _make_bar_ax(ax, values, ylabel, title):
        colors = [summary_color(v) for v in values]
        bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in values],
                      color=colors, edgecolor='black', linewidth=0.8, width=0.6)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(20, color='green',  linewidth=1.2, linestyle='--', alpha=0.7)
        ax.axhline(30, color='orange', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=22, ha='right')
        ax.set_ylabel(ylabel, fontsize=10)
        valid = [v for v in values if not np.isnan(v)]
        ax.set_ylim(0, (max(valid) * 1.25 + 5) if valid else 10)
        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a252f')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_facecolor('#f0f0f0')

    fig_s, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(18, 6))
    fig_s.patch.set_facecolor('#f5f7fa')

    _make_bar_ax(ax_d, mean_dose,
                 'Mean |% dose err|  (1-250 MeV)',
                 f'Dose Error — mean |CNN - PHITS| / PHITS\n({len(CASES)} cases)')
    _make_bar_ax(ax_m, mean_mape,
                 'Mean Flux MAPE  |y_hat-y|/y  (%)',
                 f'Flux MAPE — mean per-bin |CNN - PHITS| / PHITS\n({len(CASES)} cases)')

    legend_patches = [
        mpatches.Patch(facecolor=summary_color(10), label='< 20%'),
        mpatches.Patch(facecolor=summary_color(25), label='20-30%'),
        mpatches.Patch(facecolor=summary_color(50), label='>= 30%'),
    ]
    fig_s.legend(handles=legend_patches, fontsize=9, loc='upper right',
                 title='Threshold', title_fontsize=9)

    fig_s.suptitle('Model Summary — Dose Error & Flux MAPE across all cases',
                   fontsize=13, fontweight='bold', color='#1a252f', y=1.02)

    plt.tight_layout()
    pdf.savefig(fig_s, bbox_inches='tight', facecolor=fig_s.get_facecolor())
    plt.close(fig_s)
    print('  Summary slide done')

print(f'\nSaved → {out_path}')
