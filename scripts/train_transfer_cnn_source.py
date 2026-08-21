"""
Train a transfer head using actual v1 CNN predictions as layer-2 source spectra.

Instead of piecewise-fitted source functions, we:
  1. Run the frozen v1 model on Concrete / Steel / BPE at 5 selected thicknesses
     (using the TrackNet10 input spectrum, as layer 1 always does)
  2. Normalize each CNN output → use as the "source" for layer-2 training samples
  3. For each source, sample layer-2 (material, thickness) randomly and compute
     the target flux via the response matrix

This matches what the transfer head actually sees at inference time.

Architecture (MLP head):
  [FROZEN] Input(250,3) → v1 backbone → Dense(250)_v1
                                               ↓
                                   Dense(64, relu) → Dense(250)  [LEARNABLE]

Usage:
    ../venv/bin/python scripts/train_transfer_cnn_source.py
    ../venv/bin/python scripts/train_transfer_cnn_source.py --epochs 25 --n_per_src 10000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_spectrum
from shielding_ml.pipelines.paths import REPO_ROOT, TRACKNET10_SPECTRUM
from shielding_ml.data.constants import FLUX_SCALE

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--epochs',       type=int,   default=15)
p.add_argument('--loss',         default='mae', choices=['mae', 'mse'])
p.add_argument('--input_scale',  type=float, default=0.001)
p.add_argument('--n_per_src',    type=int,   default=10_000,
               help='Training samples per source material (spread across 5 thicknesses).')
p.add_argument('--save_dir',     default=None)
p.add_argument('--resume',       action='store_true')
args = p.parse_args()

# ── output directory ──────────────────────────────────────────────────────────
_K100S2_DIR = (REPO_ROOT / 'models'
               / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240')
SAVE_DIR = (Path(args.save_dir) if args.save_dir
            else _K100S2_DIR / 'transfer_cnn_source_mlp')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f'\n=== Transfer (CNN source, MLP head)  epochs={args.epochs} ===')
print(f'Save dir: {SAVE_DIR}\n')

# ── load & freeze base model ──────────────────────────────────────────────────
BASE_PKL = _K100S2_DIR / 'v1' / 'model.pkl'
with open(BASE_PKL, 'rb') as f:
    base_model = pickle.load(f)
base_model.trainable = False
print(f'Base model loaded  (trainable weights: {len(base_model.trainable_weights)})')

# ── build / load transfer model ───────────────────────────────────────────────
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

MODEL_PKL = SAVE_DIR / 'transfer_model.pkl'

if args.resume and MODEL_PKL.exists():
    print(f'Resuming from {MODEL_PKL}')
    with open(MODEL_PKL, 'rb') as f:
        transfer_model = pickle.load(f)
    transfer_model.compile(optimizer='adam', loss=args.loss, metrics=['mae', 'mse'])
else:
    inp      = Input(shape=(250, 3), name='input')
    base_out = base_model(inp, training=False)
    x        = Dense(64,  activation='relu', name='head_hidden')(base_out)
    head_out = Dense(250, name='correction_head')(x)
    transfer_model = Model(inputs=inp, outputs=head_out, name='k100s2_transfer_cnn_src')
    transfer_model.compile(optimizer='adam', loss=args.loss, metrics=['mae', 'mse'])

transfer_model.summary()
trainable = sum(np.prod(w.shape) for w in transfer_model.trainable_weights)
print(f'Trainable params: {trainable:,}\n')

# ── data config ───────────────────────────────────────────────────────────────
CONCRETE_DIR = '/Users/arjun/Python_Codes/Concrete/'
STEEL_DIR    = '/Users/arjun/Python_Codes/Steel/'
BPE_DIR      = '/Users/arjun/Python_Codes/BPE/'

DENSITY = {'Concrete': 2.3, 'Steel': 7.86, 'BPE': 1.04}

# 5 thicknesses per source material (spread across available range)
SRC_THICKNESSES = {
    'Concrete': [25, 50, 75, 100, 125],
    'Steel':    [20, 40, 60, 80, 100],
    'BPE':      [20, 40, 60, 80, 100],
}

# All available layer-2 thicknesses (response matrices we can use)
L2_THICKNESSES = {
    'Concrete': list(range(5,  155, 5)),   # 5,10,...,150
    'Steel':    list(range(10, 110, 10)),  # 10,20,...,100
    'BPE':      list(range(10, 110, 10)),
}
L2_MATERIALS = list(L2_THICKNESSES.keys())

# ── response matrix loader ────────────────────────────────────────────────────
_response_cache: dict = {}

def load_response(mat: str, thick: int) -> np.ndarray:
    key = (mat, thick)
    if key not in _response_cache:
        dirs = {'Concrete': CONCRETE_DIR, 'Steel': STEEL_DIR, 'BPE': BPE_DIR}
        flux = np.loadtxt(f'{dirs[mat]}{thick}cm/Neutron_Diff_Flux_Response.dat')
        R = flux.T.reshape(250, 250)         # R[j, i] = output bin j given input bin i
        _response_cache[key] = R.T / (200 * 200 * 50)   # shape (250, 250): R[i,j]
    return _response_cache[key]

# ── TrackNet10 input spectrum ─────────────────────────────────────────────────
A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))

# ── get CNN layer-1 source spectra ────────────────────────────────────────────
print('Generating CNN layer-1 source spectra ...')
src_spectra: list[tuple[str, int, np.ndarray]] = []  # (mat, thick, norm_pred)

for mat, thicks in SRC_THICKNESSES.items():
    density = DENSITY[mat]
    for thick in thicks:
        B   = np.ones(250) * thick   * args.input_scale
        B1  = np.ones(250) * density * args.input_scale
        X   = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
        log_pred = base_model.predict(X, verbose=0).flatten()
        pred = 10**log_pred * FLUX_SCALE
        s    = pred.sum()
        norm = pred / s if s > 0 else pred
        src_spectra.append((mat, thick, norm))
        print(f'  {mat} {thick}cm  sum={s:.3e}')

# ── generate training data ────────────────────────────────────────────────────
n_per_spec = args.n_per_src // len(SRC_THICKNESSES['Concrete'])  # per (src_mat, thick)
print(f'\nGenerating {n_per_spec} samples per source spectrum ...')

all_x, all_y = [], []
rng = np.random.default_rng(42)

for src_mat, src_thick, src_norm in src_spectra:
    for _ in range(n_per_spec):
        # random layer-2 material and thickness
        l2_mat   = rng.choice(L2_MATERIALS)
        l2_thick = int(rng.choice(L2_THICKNESSES[l2_mat]))
        l2_dens  = DENSITY[l2_mat]
        R        = load_response(l2_mat, l2_thick)

        # target flux = R @ src_norm  (response matrix convolution)
        target = R @ src_norm   # shape (250,)
        if target.sum() == 0:
            continue

        # model input: [source_spectrum, thickness_channel, density_channel]
        B  = np.ones(250) * l2_thick * args.input_scale
        B1 = np.ones(250) * l2_dens  * args.input_scale
        all_x.append(np.array([src_norm, B, B1]).T)
        all_y.append(np.where(target > 0, np.log10(target), 0.0))

x = np.array(all_x)  # (N, 250, 3)
y = np.array(all_y)  # (N, 250)
print(f'Dataset: {x.shape[0]} samples\n')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# ── train ─────────────────────────────────────────────────────────────────────
print(f'Training for {args.epochs} epochs ...')
history = transfer_model.fit(xtrain, ytrain, epochs=args.epochs,
                              validation_split=0.2, verbose=2)
hist = history.history

# ── save ──────────────────────────────────────────────────────────────────────
HIST_PKL = SAVE_DIR / 'history.pkl'
with open(MODEL_PKL,  'wb') as f: pickle.dump(transfer_model, f)
with open(HIST_PKL,   'wb') as f: pickle.dump(hist,           f)
np.savez(SAVE_DIR / 'testdata.npz', xtest=xtest, ytest=ytest)

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(hist['loss'],     label='train')
ax.plot(hist['val_loss'], label='val')
ax.set_xlabel('Epoch'); ax.set_ylabel(args.loss.upper())
ax.set_title('Transfer head (CNN source) — MLP 250→64→250')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / 'loss_curve.png')
plt.close()

print(f'\nSaved → {SAVE_DIR}')
print('Done.')
