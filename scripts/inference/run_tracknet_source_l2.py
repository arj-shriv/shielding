"""
Multilayer inference: v1 for layer 1, transfer heads for layer 2,
but feed TrackNet10 (not normalized CNN-L1 output) as the layer-2 source.
Layer-1 amplitude attenuation (a-factor) is still applied to the final output.

Configurations:
  baseline            — v1 L1 + v1 L2 + CNN-L1-normalized source (current default)
  transfer_mlp_c45    — v1 L1 + transfer_concrete45_mlp L2 + TrackNet10 source
  transfer_mlp_cnn    — v1 L1 + transfer_cnn_source_mlp L2 + TrackNet10 source

Prints avg_ratio, dose, and MAPE for all 5 multilayer cases.
Saves comparison PDF to inference_results/tracknet_source_l2/.
"""

import sys, re, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL, MODEL_PKL_NAME
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_ML, DATA_REF, TRACKNET10_SPECTRUM, DOSE_TABLE,
)

import pickle

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_BASE = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
OUT_DIR    = REPO_ROOT / 'inference_results' / 'tracknet_source_l2'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF    = OUT_DIR / 'comparison.pdf'

SCALE      = 0.001
BIN_SLICE  = slice(20, 150)
PHITS_CASES = sorted(PHITS_ML.glob('*.out'))

# ── Load ──────────────────────────────────────────────────────────────────────
A_SPEC    = load_spectrum(str(TRACKNET10_SPECTRUM))
VOID_FLUX, _ = load_phits(str(DATA_REF / 'No_Shielding.out'))
_R_ALL    = _build_r_all(DOSE_TABLE)

def dose_fn(flux): return dose_midpoint(flux, _R_ALL, excl_bin0=True)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

model_v1   = load_pkl(MODEL_BASE / 'v1' / MODEL_PKL_NAME)
model_c45  = load_pkl(MODEL_BASE / 'transfer_concrete45_mlp' / 'transfer_model.pkl')
model_cnn  = load_pkl(MODEL_BASE / 'transfer_cnn_source_mlp' / 'transfer_model.pkl')
_conv_pkl  = MODEL_BASE / 'transfer_conv_head' / 'transfer_model.pkl'
model_conv = load_pkl(_conv_pkl) if _conv_pkl.exists() else None
print("Models loaded.")

# ── Inference helpers ─────────────────────────────────────────────────────────
def _predict(model, src, mat, thick):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * SCALE
    B1 = np.ones(250) * density * SCALE
    X  = np.array([src, B, B1]).T.reshape(1, 250, 3)
    return 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE


def baseline(layers):
    """v1 for both layers, CNN-L1 normalized output as layer-2 source."""
    mat1, thick1 = layers[0]
    mat2, thick2 = layers[1]
    pred_L1  = _predict(model_v1, A_SPEC, mat1, thick1)
    a        = pred_L1[BIN_SLICE].sum() / VOID_FLUX[BIN_SLICE].sum()
    norm_L1  = pred_L1 / pred_L1.sum()
    pred_L2  = _predict(model_v1, norm_L1, mat2, thick2)
    return pred_L2 * a


def tracknet_source_l2(model_l2, layers):
    """v1 for layer 1 (amplitude only), model_l2 for layer 2 with TrackNet10 source."""
    mat1, thick1 = layers[0]
    mat2, thick2 = layers[1]
    pred_L1 = _predict(model_v1, A_SPEC, mat1, thick1)
    a       = pred_L1[BIN_SLICE].sum() / VOID_FLUX[BIN_SLICE].sum()
    pred_L2 = _predict(model_l2, A_SPEC, mat2, thick2)   # TrackNet10 as source
    return pred_L2 * a


def avg_ratio(pred, phits):
    mask = phits[BIN_SLICE] > 0
    return float(np.mean(pred[BIN_SLICE][mask] / phits[BIN_SLICE][mask]))


def mape(pred, phits):
    mask = phits[BIN_SLICE] > 0
    return float(np.mean(np.abs(pred[BIN_SLICE][mask] - phits[BIN_SLICE][mask])
                         / phits[BIN_SLICE][mask]) * 100)


def parse_layers(path):
    name = os.path.basename(path).replace('.out', '')
    return [(m, int(t)) for m, t in re.findall(r'([A-Za-z]+)_(\d+)cm', name)]


# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS = [
    ('transfer_mlp_cnn', 'seagreen', lambda layers: tracknet_source_l2(model_cnn, layers)),
]

# ── Run all cases ─────────────────────────────────────────────────────────────
print(f"\n{'Case':<38} {'Config':<22} {'avg_ratio':>10} {'MAPE%':>8} {'dose CNN':>12} {'dose err%':>10}")
print('-' * 104)

results = {}
with PdfPages(OUT_PDF) as pdf:
    for p in PHITS_CASES:
        layers = parse_layers(str(p))
        if not layers or any(m not in MATERIALS for m, _ in layers):
            continue

        phits_flux, _ = load_phits(str(p))
        if phits_flux is None:
            continue

        case_lbl    = os.path.basename(str(p)).replace('.out', '').replace('_', ' ')
        phits_dose  = dose_fn(phits_flux)
        results[case_lbl] = {}

        fig = plt.figure(figsize=(14, 9))
        gs  = plt.GridSpec(2, 2, height_ratios=[2.2, 1], hspace=0.45, figure=fig)
        ax1    = fig.add_subplot(gs[0, 0])
        ax2    = fig.add_subplot(gs[0, 1])
        ax_tbl = fig.add_subplot(gs[1, :])
        ax_tbl.axis('off')

        ax1.semilogy(E_ALL, phits_flux, 'k-', lw=2, label='PHITS', zorder=5)
        ax2.axhline(1.0, color='k', ls='--', lw=1)
        ax2.axhspan(0.8, 1.2, alpha=0.08, color='green')

        dose_rows = []
        for cfg_name, color, infer_fn in CONFIGS:
            pred = infer_fn(layers)
            ar   = avg_ratio(pred, phits_flux)
            mp   = mape(pred, phits_flux)
            d    = dose_fn(pred)
            pct  = 100 * (d - phits_dose) / phits_dose
            results[case_lbl][cfg_name] = dict(avg_ratio=ar, mape=mp, dose=d, dose_pct=pct)
            dose_rows.append((cfg_name, ar, mp, d, pct, color))

            print(f'{case_lbl:<38} {cfg_name:<22} {ar:>10.4f} {mp:>8.2f} {d:>12.3e} {pct:>+8.1f}%')

            ratio = np.where(phits_flux > 0, pred / phits_flux, np.nan)
            ax1.semilogy(E_ALL, pred, color=color, lw=1.4, ls='--',
                         label=f'{cfg_name}  (avg_ratio={ar:.3f})')
            ax2.plot(E_ALL[20:150], ratio[20:150], color=color, lw=1.2, label=cfg_name)

        ax1.set_xlim(0, 250); ax1.set_ylim(bottom=1e-7)
        ax1.set_xlabel('Energy (MeV)'); ax1.set_ylabel('Flux [n/cm²/source]')
        _dose_pct = dose_rows[0][4] if dose_rows else None
        ax1.text(0.5, 1.10, case_lbl, transform=ax1.transAxes,
                 ha='center', fontsize=10, fontweight='bold', clip_on=False)
        if _dose_pct is not None:
            ax1.text(0.5, 1.025, f'dose err {_dose_pct:+.1f}%',
                     transform=ax1.transAxes, ha='center', fontsize=9.5,
                     fontweight='bold', color='crimson', clip_on=False)
        ax1.legend(fontsize=8)

        ax2.set_xlim(20, 150)
        ax2.set_xlabel('Energy (MeV)'); ax2.set_ylabel('CNN / PHITS')
        ax2.set_title('Ratio (bins 20–150)')
        ax2.legend(fontsize=8)
        valid = np.concatenate([np.where(phits_flux[BIN_SLICE] > 0,
                                         infer_fn(layers)[BIN_SLICE] / phits_flux[BIN_SLICE],
                                         np.nan) for _, _, infer_fn in CONFIGS])
        ax2.set_ylim(0, min(np.nanpercentile(valid, 98) * 1.2, 15))

        # ── dose table ────────────────────────────────────────────────────
        def _ec(pct):
            v = abs(pct)
            if v < 20:  return (0.55, 0.86, 0.60, 0.7)
            if v < 50:  return (1.00, 0.93, 0.45, 0.7)
            return (0.95, 0.45, 0.45, 0.7)

        col_labels = ['Config', 'avg_ratio', 'MAPE %', 'CNN dose (pSv/src)', 'Dose error %', f'PHITS dose (pSv/src)']
        tbl_data   = [[n, f'{ar:.4f}', f'{mp:.1f}%', f'{d:.3e}', f'{pct:+.1f}%', f'{phits_dose:.3e}']
                      for n, ar, mp, d, pct, _ in dose_rows]
        cell_colors = [['white', 'white', 'white', 'white', _ec(pct), '#d0e8ff']
                       for _, _, _, _, pct, _ in dose_rows]

        tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels,
                           cellColours=cell_colors, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#2c3e50')
            tbl[0, j].set_text_props(color='white', fontweight='bold')

        fig.suptitle(f'TrackNet10 as layer-2 source  |  {case_lbl}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print()

print(f'\nSaved → {OUT_PDF}')

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n── Summary: avg_ratio ──────────────────────────────────────────────')
header = f"{'Case':<38}" + ''.join(f'{n:>22}' for n, _, _ in CONFIGS)
print(header)
for case, cfg_res in results.items():
    row = f'{case:<38}' + ''.join(f'{cfg_res[n]["avg_ratio"]:>22.4f}' for n, _, _ in CONFIGS)
    print(row)

print('\n── Summary: MAPE% ──────────────────────────────────────────────────')
print(header)
for case, cfg_res in results.items():
    row = f'{case:<38}' + ''.join(f'{cfg_res[n]["mape"]:>22.2f}' for n, _, _ in CONFIGS)
    print(row)

print('\n── Summary: dose error % ───────────────────────────────────────────')
print(header)
for case, cfg_res in results.items():
    row = f'{case:<38}' + ''.join(f'{cfg_res[n]["dose_pct"]:>+22.1f}' for n, _, _ in CONFIGS)
    print(row)
