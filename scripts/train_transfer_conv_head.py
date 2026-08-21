"""
Train a convolutional transfer head on top of the frozen v1 conv backbone.

Architecture:
  [FROZEN]  Input(250,3) → Conv1D_0 → MaxPool → Conv1D_1 → MaxPool
                         → Conv1D_2 → Conv1D_3   ← cut here (16, 64)
  [NEW]     Conv1D(50, k=50, s=1, same) → Conv1D(10, k=10, s=2, same)
                         → Flatten(80) → Dense(250)
  New trainable params: 185,310

Training data: same as transfer_cnn_source_mlp — actual v1 CNN layer-1
predictions (Concrete/Steel/BPE at 5 thicknesses) used as layer-2 sources;
targets computed via response-matrix convolution.

Usage:
    ../venv/bin/python scripts/train_transfer_conv_head.py
    ../venv/bin/python scripts/train_transfer_conv_head.py --epochs 25
"""

from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_spectrum
from shielding_ml.pipelines.paths import REPO_ROOT, TRACKNET10_SPECTRUM
from shielding_ml.data.constants import FLUX_SCALE

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--epochs',      type=int,   default=15)
p.add_argument('--loss',        default='mae', choices=['mae', 'mse'])
p.add_argument('--input_scale', type=float, default=0.001)
p.add_argument('--n_per_src',   type=int,   default=10_000,
               help='Training samples per source material (spread across 5 thicknesses).')
p.add_argument('--save_dir',    default=None)
args = p.parse_args()

_K100S2_DIR = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
SAVE_DIR = Path(args.save_dir) if args.save_dir else _K100S2_DIR / 'transfer_conv_head'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f'\n=== Transfer (CNN source, Conv head)  epochs={args.epochs} ===')
print(f'Save dir: {SAVE_DIR}\n')

# ── load v1 and extract frozen conv backbone ──────────────────────────────────
BASE_PKL = _K100S2_DIR / 'v1' / 'model.pkl'
with open(BASE_PKL, 'rb') as f:
    v1 = pickle.load(f)

# Cut the v1 backbone after the last Conv1D (Conv1D_3) → output (None, 16, 64).
# Use name matching so layer ordering doesn't matter.
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Flatten, Dense

backbone_layers = []
for layer in v1.layers:
    lname = layer.name.lower()
    if 'conv1d' in lname or 'max_pool' in lname or 'pooling' in lname:
        backbone_layers.append(layer)
    elif backbone_layers:
        break  # past the conv/pool block → stop

inp = Input(shape=(250, 3), name='input')
x = inp
for layer in backbone_layers:
    layer.trainable = False
    x = layer(x, training=False)

frozen_params = sum(np.prod(w.shape) for w in
                    [w for layer in backbone_layers for w in layer.weights])
print(f'Frozen backbone layers: {[l.name for l in backbone_layers]}')
print(f'Frozen backbone output shape: {x.shape}  (frozen params: {frozen_params:,})')

# ── new trainable conv head ───────────────────────────────────────────────────
x = Conv1D(50, kernel_size=50, strides=1, padding='same',
           activation='relu', name='conv_head_0')(x)   # → (16, 50)
x = Conv1D(10, kernel_size=10, strides=2, padding='same',
           name='conv_head_1')(x)                       # → (8, 10)
x = Flatten(name='head_flatten')(x)                     # → 80
out = Dense(250, name='head_output')(x)                 # → 250

transfer_model = Model(inputs=inp, outputs=out, name='k100s2_transfer_conv')
transfer_model.compile(optimizer='adam', loss=args.loss, metrics=['mae', 'mse'])
transfer_model.summary()

trainable = sum(np.prod(w.shape) for w in transfer_model.trainable_weights)
print(f'\nNew trainable params: {trainable:,}\n')

# ── data config (same as train_transfer_cnn_source.py) ───────────────────────
CONCRETE_DIR = '/Users/arjun/Python_Codes/Concrete/'
STEEL_DIR    = '/Users/arjun/Python_Codes/Steel/'
BPE_DIR      = '/Users/arjun/Python_Codes/BPE/'

DENSITY = {'Concrete': 2.3, 'Steel': 7.86, 'BPE': 1.04}

SRC_THICKNESSES = {
    'Concrete': [25, 50, 75, 100, 125],
    'Steel':    [20, 40, 60, 80, 100],
    'BPE':      [20, 40, 60, 80, 100],
}
L2_THICKNESSES = {
    'Concrete': list(range(5,  155, 5)),
    'Steel':    list(range(10, 110, 10)),
    'BPE':      list(range(10, 110, 10)),
}
L2_MATERIALS = list(L2_THICKNESSES.keys())

_response_cache: dict = {}
def load_response(mat, thick):
    key = (mat, thick)
    if key not in _response_cache:
        dirs = {'Concrete': CONCRETE_DIR, 'Steel': STEEL_DIR, 'BPE': BPE_DIR}
        flux = np.loadtxt(f'{dirs[mat]}{thick}cm/Neutron_Diff_Flux_Response.dat')
        R = flux.T.reshape(250, 250)
        _response_cache[key] = R.T / (200 * 200 * 50)
    return _response_cache[key]

A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))

# ── generate CNN layer-1 source spectra ──────────────────────────────────────
print('Generating CNN layer-1 source spectra ...')
src_spectra: list[tuple[str, int, np.ndarray]] = []

for mat, thicks in SRC_THICKNESSES.items():
    density = DENSITY[mat]
    for thick in thicks:
        B   = np.ones(250) * thick   * args.input_scale
        B1  = np.ones(250) * density * args.input_scale
        X   = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
        log_pred = v1.predict(X, verbose=0).flatten()
        pred = 10**log_pred * FLUX_SCALE
        s    = pred.sum()
        norm = pred / s if s > 0 else pred
        src_spectra.append((mat, thick, norm))
        print(f'  {mat} {thick}cm  sum={s:.3e}')

# ── generate training data ────────────────────────────────────────────────────
n_per_spec = args.n_per_src // 5
print(f'\nGenerating {n_per_spec} samples per source spectrum ({len(src_spectra)} spectra) ...')

all_x, all_y = [], []
rng = np.random.default_rng(42)

for src_mat, src_thick, src_norm in src_spectra:
    for _ in range(n_per_spec):
        l2_mat   = rng.choice(L2_MATERIALS)
        l2_thick = int(rng.choice(L2_THICKNESSES[l2_mat]))
        l2_dens  = DENSITY[l2_mat]
        R        = load_response(l2_mat, l2_thick)
        target   = R @ src_norm
        if target.sum() == 0:
            continue
        B  = np.ones(250) * l2_thick * args.input_scale
        B1 = np.ones(250) * l2_dens  * args.input_scale
        all_x.append(np.array([src_norm, B, B1]).T)
        all_y.append(np.where(target > 0, np.log10(target), 0.0))

x = np.array(all_x)
y = np.array(all_y)
print(f'Dataset: {x.shape[0]} samples\n')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# ── train ─────────────────────────────────────────────────────────────────────
print(f'Training for {args.epochs} epochs ...')
history = transfer_model.fit(xtrain, ytrain, epochs=args.epochs,
                              validation_split=0.2, batch_size=64, verbose=2)
hist = history.history

# ── save ──────────────────────────────────────────────────────────────────────
MODEL_PKL = SAVE_DIR / 'transfer_model.pkl'
with open(MODEL_PKL,              'wb') as f: pickle.dump(transfer_model, f)
with open(SAVE_DIR / 'history.pkl', 'wb') as f: pickle.dump(hist, f)
np.savez(SAVE_DIR / 'testdata.npz', xtest=xtest, ytest=ytest)

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(hist['loss'],     label='train')
ax.plot(hist['val_loss'], label='val')
ax.set_xlabel('Epoch'); ax.set_ylabel(args.loss.upper())
ax.set_title('Transfer head (CNN source) — Conv 50×50 → 10×10 → Dense 250')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / 'loss_curve.png')
plt.close()

print(f'\nSaved → {SAVE_DIR}')
print(f'Final val MAE: {hist["val_mae"][-1]:.4f}')
