"""
Transfer learning: train a single Dense(250) correction head on top of the frozen k100s2 v1.

Full architecture
-----------------
  [FROZEN — entire v1 model]
  Input(250,3) → Conv1D(100) → MaxPool → Conv1D(25) → MaxPool
               → Conv1D(11) → Conv1D(3) → Flatten
               → Dense(256) → Dense(64) → Dense(250)_v1
                                                   ↓
                                    [LEARNABLE — 62,750 params]
                                         Dense(250)_head

Routing at inference time
-------------------------
  Single-layer / Layer 1:   Dense(250)_v1   (original model, unchanged)
  Layer 2+ in multilayer:   Dense(250)_v1 → Dense(250)_head

Usage
-----
    ../venv/bin/python scripts/train_transfer_head.py
    ../venv/bin/python scripts/train_transfer_head.py --sources concrete25 steel27 --epochs 15
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

from shielding_ml.data.source_spectra import SOURCE_FUNCTIONS
from shielding_ml.pipelines.paths import REPO_ROOT

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description='Train Dense(250) correction head on frozen k100s2 v1.')
p.add_argument('--sources',     nargs='+', default=['concrete25', 'steel27', 'bpe25'],
               choices=list(SOURCE_FUNCTIONS.keys()),
               help='Source spectra to train on (default: all three shielded sources).')
p.add_argument('--epochs',      type=int,   default=25)
p.add_argument('--loss',        default='mae', choices=['mae', 'mse'])
p.add_argument('--input_scale', type=float, default=0.001)
p.add_argument('--n_bins_min',  type=int,   default=50)
p.add_argument('--n_bins_max',  type=int,   default=240)
p.add_argument('--save_dir',    default=None)
p.add_argument('--resume',      action='store_true',
               help='Load existing transfer_model.pkl and continue training.')
p.add_argument('--arch',        default='linear', choices=['linear', 'mlp'],
               help='Head architecture: linear=250→250, mlp=250→64→250 (default: linear).')
args = p.parse_args()

# ── output directory ──────────────────────────────────────────────────────────
_K100S2_BASE = (Path(__file__).resolve().parent.parent
                / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240')
src_tag  = '_'.join(args.sources)
arch_tag = 'linear' if args.arch == 'linear' else 'mlp'
_default_dir = _K100S2_BASE / f'transfer_{src_tag}_{arch_tag}'
SAVE_DIR = Path(args.save_dir) if args.save_dir else _default_dir
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f'\n=== Transfer head  sources={args.sources}  epochs={args.epochs} ===')
print(f'Save dir: {SAVE_DIR}\n')

# ── load & freeze base model ──────────────────────────────────────────────────
BASE_PKL = (REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
            / 'v1' / 'model.pkl')

with open(BASE_PKL, 'rb') as f:
    base_model = pickle.load(f)

base_model.trainable = False
print(f'Base model loaded: {BASE_PKL}')
print(f'Trainable weights after freeze: {len(base_model.trainable_weights)}  (should be 0)\n')

# ── build transfer model ──────────────────────────────────────────────────────
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
    base_out = base_model(inp, training=False)   # (batch, 250) frozen
    if args.arch == 'mlp':
        x        = Dense(64,  activation='relu', name='head_hidden')(base_out)
        head_out = Dense(250, name='correction_head')(x)
        model_name = 'k100s2_transfer_mlp'
    else:
        head_out   = Dense(250, name='correction_head')(base_out)
        model_name = 'k100s2_transfer_linear'
    transfer_model = Model(inputs=inp, outputs=head_out, name=model_name)
    transfer_model.compile(optimizer='adam', loss=args.loss, metrics=['mae', 'mse'])

transfer_model.summary()

trainable = sum(np.prod(w.shape) for w in transfer_model.trainable_weights)
print(f'\nTrainable params: {trainable:,}')

# ── data directories ──────────────────────────────────────────────────────────
CONCRETE_DIR = '/Users/arjun/Python_Codes/Concrete/'
STEEL_DIR    = '/Users/arjun/Python_Codes/Steel/'
BPE_DIR      = '/Users/arjun/Python_Codes/BPE/'

DENSITY            = {'Concrete': 2.3, 'Steel': 7.86, 'BPE': 1.04}
Thickness_concrete = np.array([5.0] + list(np.linspace(10, 150, 15)))
Thickness_steel    = np.linspace(10, 100, 10)
Thickness_BPE      = np.linspace(10, 100, 10)


def load_response(mat_dir: str, thicknesses: np.ndarray) -> np.ndarray:
    Y1 = []
    for t in thicknesses:
        flux = np.loadtxt(f'{mat_dir}{int(t)}cm/Neutron_Diff_Flux_Response.dat')
        flux = flux.T
        for j in range(250):
            Y1.append(list(flux[j]))
    Y1 = np.array(Y1).reshape(len(thicknesses), 250, 250)
    Y2 = np.array([Y1[i].T for i in range(len(thicknesses))]).T
    return Y2 / (200 * 200 * 50)


def generate_output_flux(Y, thicknesses, density, source_fn):
    E_MeV   = np.linspace(0.5, 249.5, 250)
    weights = source_fn(E_MeV)
    X_new, Y_new = [], []
    for k in range(len(thicknesses)):
        for _ in range(10_000):
            A   = np.zeros(250)
            w1  = np.random.randint(args.n_bins_min, args.n_bins_max + 1)
            idx = np.random.randint(0, 249, size=w1)
            for j in idx:
                A[j] = weights[j]
            A[A == 0] = 1e-7
            A /= A.sum()
            B  = np.ones(250) * thicknesses[k] * args.input_scale
            B1 = np.ones(250) * density         * args.input_scale
            X_new.append(np.array([A, B, B1]).T)
            S = sum(Y[j, :, k] * A[j] for j in range(250))
            Y_new.append(np.where(S != 0, np.log10(S), 0.0))
    return np.array(X_new), np.array(Y_new)


# ── generate training data ────────────────────────────────────────────────────
print('Loading response matrices ...')
Y_c = load_response(CONCRETE_DIR, Thickness_concrete)
Y_s = load_response(STEEL_DIR,    Thickness_steel)
Y_b = load_response(BPE_DIR,      Thickness_BPE)

all_x, all_y = [], []
for src_name in args.sources:
    src_fn = SOURCE_FUNCTIONS[src_name]
    print(f'Generating data — source: {src_name} ...')
    xc, yc = generate_output_flux(Y_c, Thickness_concrete, DENSITY['Concrete'], src_fn)
    xs, ys = generate_output_flux(Y_s, Thickness_steel,    DENSITY['Steel'],    src_fn)
    xb, yb = generate_output_flux(Y_b, Thickness_BPE,      DENSITY['BPE'],      src_fn)
    all_x.append(np.concatenate([xc, xs, xb]))
    all_y.append(np.concatenate([yc, ys, yb]))

x = np.concatenate(all_x).reshape(-1, 250, 3)
y = np.concatenate(all_y).reshape(-1, 250)
print(f'Dataset: {x.shape[0]} samples\n')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# ── train ─────────────────────────────────────────────────────────────────────
print(f'Training for {args.epochs} epochs ...')
history = transfer_model.fit(xtrain, ytrain, epochs=args.epochs,
                              validation_split=0.3, verbose=2)
hist = history.history

# ── save ──────────────────────────────────────────────────────────────────────
HIST_PKL  = SAVE_DIR / 'history.pkl'
with open(MODEL_PKL,  'wb') as f: pickle.dump(transfer_model, f)
with open(HIST_PKL,   'wb') as f: pickle.dump(hist,           f)
np.savez(SAVE_DIR / 'testdata.npz', xtest=xtest, ytest=ytest)
print(f'\nSaved → {SAVE_DIR}')

# ── loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(hist['loss'],     label='train')
ax.plot(hist['val_loss'], label='val')
ax.set_xlabel('Epoch'); ax.set_ylabel(args.loss.upper())
ax.set_title('Transfer head — sources: ' + ', '.join(args.sources))
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / 'loss_curve.png')
plt.close()

print('Done.')
