"""
Test: multilayer inference where the normalized layer-1 output is rescaled
so its sum matches the TrackNet10 source sum (instead of normalizing to 1).

Prints avg_ratio for all 5 multilayer PHITS cases, comparing:
  baseline  — normalize to sum=1  (current behaviour)
  renorm    — normalize to sum=A_SPEC.sum()
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, MODEL_PKL_NAME
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import REPO_ROOT, PHITS_ML, DATA_REF, TRACKNET10_SPECTRUM

import pickle, re

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240' / 'v1'
SCALE      = 0.001
VOID_FILE  = DATA_REF / 'No_Shielding.out'
BIN_SLICE  = slice(20, 150)

PHITS_CASES = sorted(PHITS_ML.glob('*.out'))

# ── Load ──────────────────────────────────────────────────────────────────────
A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
A_SUM  = A_SPEC.sum()
print(f'TrackNet10 sum: {A_SUM:.6f}')

VOID_FLUX, _ = load_phits(str(VOID_FILE))

model = load_model(str(MODEL_DIR / MODEL_PKL_NAME))
print(f'Model loaded: {MODEL_DIR}')


# ── Inference helpers ─────────────────────────────────────────────────────────
def _predict(spec, mat, thick):
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * SCALE
    B1 = np.ones(250) * density * SCALE
    X  = np.array([spec, B, B1]).T.reshape(1, 250, 3)
    return 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE


def multilayer(layers, renorm=False):
    """
    renorm=False  →  normalize each layer output to sum=1 before feeding next
    renorm=True   →  normalize to sum=A_SPEC.sum() before feeding next
    """
    spec = A_SPEC.copy()
    cumulative_scale = 1.0

    for i, (mat, thick) in enumerate(layers):
        pred = _predict(spec, mat, thick)

        if i == 0:
            ref = VOID_FLUX[BIN_SLICE].sum()
            cumulative_scale = pred[BIN_SLICE].sum() / ref if ref > 0 else 1.0

        if i < len(layers) - 1:
            s = pred.sum()
            if renorm:
                spec = pred / s * A_SUM if s > 0 else pred
            else:
                spec = pred / s if s > 0 else pred

    return pred * cumulative_scale


def avg_ratio(pred, phits):
    mask = phits[BIN_SLICE] > 0
    return float(np.mean(pred[BIN_SLICE][mask] / phits[BIN_SLICE][mask]))


def parse_layers(path):
    name = path.stem
    return [(m, int(t)) for m, t in re.findall(r'([A-Za-z]+)_(\d+)cm', name)]


# ── Run ───────────────────────────────────────────────────────────────────────
print(f'\n{"Case":<40}  {"baseline":>10}  {"renorm":>10}  {"delta":>8}')
print('-' * 74)

for p in PHITS_CASES:
    layers = parse_layers(p)
    if not layers:
        continue
    if any(m not in MATERIALS for m, _ in layers):
        continue

    phits_flux, _ = load_phits(str(p))
    if phits_flux is None:
        continue

    case_lbl = p.stem.replace('_', ' ')

    pred_base  = multilayer(layers, renorm=False)
    pred_renorm = multilayer(layers, renorm=True)

    ar_base   = avg_ratio(pred_base,   phits_flux)
    ar_renorm = avg_ratio(pred_renorm, phits_flux)
    delta     = ar_renorm - ar_base

    print(f'{case_lbl:<40}  {ar_base:>10.4f}  {ar_renorm:>10.4f}  {delta:>+8.4f}')

print()
print('avg_ratio ideal = 1.0')
print('renorm: layer-1 output sum is rescaled to match TrackNet10 sum before layer-2 input')
