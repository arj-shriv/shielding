"""
Compare three layer-2 inference approaches for Concrete 45cm + Steel 33cm:

  (baseline)  k100s2-v1  with TrackNet10 as layer-1 source         (current default)
  (B)         k100s2-v1  with actual PHITS Concrete 45cm flux as source for layer 2
              — this isolates layer-2 CNN error by bypassing the layer-1 CNN entirely
  (C)         bi-model: k100s2-v1 for layer 1, v_concrete25 for layer 2
              — source-matched model since v_concrete25 was trained on concrete-shielded spectra

Usage:
    ../venv/bin/python scripts/inference/compare_source_variants.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_phits_flux, load_spectrum
from shielding_ml.pipelines.paths import REPO_ROOT, TRACKNET10_SPECTRUM
from shielding_ml.data.constants import FLUX_SCALE

# ── paths ─────────────────────────────────────────────────────────────────────
K100_DIR   = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
PHITS_ML   = REPO_ROOT / 'data/raw/phits/multilayer/Concrete_45cm_Steel_33cm.out'
PHITS_C45  = REPO_ROOT / 'data/raw/phits/high_energy/Concrete_45cm_1.out'
VOID_FILE  = REPO_ROOT / 'data/reference/No_Shielding.out'

SCALE = 0.001
BIN_SLICE = slice(20, 150)   # same as run_multilayer.py

# ── load models ───────────────────────────────────────────────────────────────
with open(K100_DIR / 'v1' / 'model.pkl', 'rb') as f:
    model_v1 = pickle.load(f)
with open(K100_DIR / 'v_concrete25' / 'model.pkl', 'rb') as f:
    model_c25 = pickle.load(f)

# ── load reference data ───────────────────────────────────────────────────────
phits_ml  = load_phits_flux(PHITS_ML)    # Concrete 45 + Steel 33 combined PHITS
phits_c45 = load_phits_flux(PHITS_C45)  # single-layer Concrete 45 PHITS
void_flux = load_phits_flux(VOID_FILE)
A_SPEC    = load_spectrum(str(TRACKNET10_SPECTRUM))  # normalized TrackNet10


def _predict(model, source_norm, mat_thick, mat_dens):
    """Single CNN forward pass. Returns absolute-flux prediction."""
    B  = np.ones(250) * mat_thick * SCALE
    B1 = np.ones(250) * mat_dens  * SCALE
    X  = np.array([source_norm, B, B1]).T.reshape(1, 250, 3)
    log_pred = model.predict(X, verbose=0).flatten()
    return 10**log_pred * FLUX_SCALE


# ── (baseline) k100s2-v1 with TrackNet10 source ───────────────────────────────
# Layer 1: Concrete 45cm
pred_L1_v1 = _predict(model_v1, A_SPEC, 45, 2.3)
a_v1       = pred_L1_v1[BIN_SLICE].sum() / void_flux[BIN_SLICE].sum()

norm_L1_v1 = pred_L1_v1 / pred_L1_v1.sum()
# Layer 2: Steel 33cm  (no b-factor — last layer, same as run_multilayer.py)
pred_L2_v1 = _predict(model_v1, norm_L1_v1, 33, 7.86)
result_baseline = pred_L2_v1 * a_v1

# ── (B) k100s2-v1, but use PHITS Concrete 45cm flux as layer-2 source ─────────
# Amplitude from real PHITS layer-1 flux
a_phits   = phits_c45[BIN_SLICE].sum() / void_flux[BIN_SLICE].sum()
norm_c45  = phits_c45 / phits_c45.sum()
pred_L2_B = _predict(model_v1, norm_c45, 33, 7.86)
result_B  = pred_L2_B * a_phits

# ── (C) bi-model: v1 for layer 1, v_concrete25 for layer 2 ────────────────────
# Layer 1 same as baseline
pred_L2_C = _predict(model_c25, norm_L1_v1, 33, 7.86)
result_C  = pred_L2_C * a_v1

# ── print avg_ratio summary ───────────────────────────────────────────────────
E = np.linspace(0.5, 249.5, 250)
mask = (phits_ml[BIN_SLICE] > 0)

def avg_ratio(pred):
    r = pred[BIN_SLICE][mask] / phits_ml[BIN_SLICE][mask]
    return float(np.mean(r))

print(f"amplitude factors:")
print(f"  a_v1   (CNN layer-1 / void) : {a_v1:.4f}")
print(f"  a_phits (PHITS C45 / void)  : {a_phits:.4f}")
print(f"\nraw L2 pred sums (bins 20:150):")
print(f"  baseline L2 pred sum        : {pred_L2_v1[BIN_SLICE].sum():.4e}")
print(f"  (B)      L2 pred sum        : {pred_L2_B[BIN_SLICE].sum():.4e}")
print(f"  (C)      L2 pred sum        : {pred_L2_C[BIN_SLICE].sum():.4e}")
print(f"  PHITS multilayer sum        : {phits_ml[BIN_SLICE].sum():.4e}")
print(f"\navg_ratio  CNN/PHITS  (bins 20:150):")
print(f"  baseline (TrackNet10 src)    : {avg_ratio(result_baseline):.4f}")
print(f"  (B) PHITS C45 src            : {avg_ratio(result_B):.4f}")
print(f"  (C) bi-model v_concrete25 L2 : {avg_ratio(result_C):.4f}")
print(f"  (ideal = 1.0)")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

ax.plot(E, phits_ml,       color='black',   lw=2,   label='PHITS  Concrete 45 + Steel 33')
ax.plot(E, result_baseline, color='steelblue', lw=1.5, ls='--',
        label=f'k100 v1  |  TrackNet10 src  (avg_ratio={avg_ratio(result_baseline):.2f})')
ax.plot(E, result_B,        color='tomato',    lw=1.5, ls='-.',
        label=f'(B) k100 v1  |  PHITS C45 src  (avg_ratio={avg_ratio(result_B):.2f})')
ax.plot(E, result_C,        color='seagreen',  lw=1.5, ls=':',
        label=f'(C) bi-model v_concrete25 L2  (avg_ratio={avg_ratio(result_C):.2f})')

ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Neutron flux [n/cm²/source]')
ax.set_title('Concrete 45cm + Steel 33cm  —  source variant comparison')
ax.set_yscale('log')
ax.set_xlim(0, 250)
ax.legend(fontsize=8)
ax.axvspan(20, 150, alpha=0.05, color='gray', label='_avg_ratio range')
plt.tight_layout()

OUT = REPO_ROOT / 'inference_results' / 'source_variant_comparison.png'
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT)
print(f'\nSaved → {OUT}')
