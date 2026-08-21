"""
Train the k100s2 architecture with a selectable source spectrum.

Architecture: Conv1D(64,100)→MaxPool→Conv1D(64,25)→MaxPool→Conv1D(64,11)→Conv1D(64,3)
              →Flatten→Dense(256)→Dense(64)→Dense(250)

Usage
-----
    ../venv/bin/python scripts/train_k100s2_multisrc.py --source concrete25
    ../venv/bin/python scripts/train_k100s2_multisrc.py --source steel27
    ../venv/bin/python scripts/train_k100s2_multisrc.py --source bpe25
    ../venv/bin/python scripts/train_k100s2_multisrc.py --source tracknet10  # baseline

All outputs go to /Users/arjun/Python_Codes/<YYYYMMDD>/k100s2_<source>_src/
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

# ── repo on path ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.source_spectra import SOURCE_FUNCTIONS

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description='Train k100s2 model with selectable source.')
p.add_argument('--source',      required=True, choices=list(SOURCE_FUNCTIONS.keys()),
               help='Source spectrum to use for training data generation.')
p.add_argument('--epochs',      type=int,   default=25)
p.add_argument('--loss',        default='mae', choices=['mae', 'mse'])
p.add_argument('--input_scale', type=float, default=0.001)
p.add_argument('--n_bins_min',  type=int,   default=50)
p.add_argument('--n_bins_max',  type=int,   default=240)
p.add_argument('--save_dir',    default=None,
               help='Override output directory (default: /Users/arjun/Python_Codes/<date>/k100s2_<source>_src).')
args = p.parse_args()

# ── output directory ──────────────────────────────────────────────────────────
date_str = datetime.now().strftime('%Y%m%d')
SAVE_DIR = (Path(args.save_dir) if args.save_dir
            else Path('/Users/arjun/Python_Codes') / date_str / f'k100s2_{args.source}_src')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f'\n=== k100s2  source={args.source}  epochs={args.epochs} ===')
print(f'Save dir: {SAVE_DIR}\n')

MODEL_PKL   = SAVE_DIR / 'model.pkl'
HISTORY_PKL = SAVE_DIR / 'history.pkl'
TESTDATA    = SAVE_DIR / 'testdata.npz'

# ── response-matrix data directories ─────────────────────────────────────────
CONCRETE_DIR = '/Users/arjun/Python_Codes/Concrete/'
STEEL_DIR    = '/Users/arjun/Python_Codes/Steel/'
BPE_DIR      = '/Users/arjun/Python_Codes/BPE/'

DENSITY = {'Concrete': 2.3, 'Steel': 7.86, 'BPE': 1.04}

Thickness_concrete = np.array([5.0] + list(np.linspace(10, 150, 15)))  # 16 values
Thickness_steel    = np.linspace(10, 100, 10)
Thickness_BPE      = np.linspace(10, 100, 10)


# ── response matrix loader ────────────────────────────────────────────────────

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


# ── training data generator ───────────────────────────────────────────────────

source_fn = SOURCE_FUNCTIONS[args.source]

def generate_output_flux(Y: np.ndarray, thicknesses: np.ndarray,
                         density: float) -> tuple[np.ndarray, np.ndarray]:
    E_MeV   = np.linspace(0.5, 249.5, 250)
    weights = source_fn(E_MeV)   # one noisy source realization per call
    X_new, Y_new = [], []

    for k in range(len(thicknesses)):
        for _ in range(10_000):
            A  = np.zeros(250)
            w1 = np.random.randint(args.n_bins_min, args.n_bins_max + 1)
            idx = np.random.randint(0, 249, size=w1)
            for j in idx:
                A[j] = weights[j]
            A[A == 0] = 1e-7
            A /= A.sum()

            B  = np.ones(250) * thicknesses[k] * args.input_scale
            B1 = np.ones(250) * density         * args.input_scale
            X_new.append(np.array([A, B, B1]).T)

            S  = sum(Y[j, :, k] * A[j] for j in range(250))
            Y_new.append(np.where(S != 0, np.log10(S), 0.0))

    return np.array(X_new), np.array(Y_new)


# ── build / train ─────────────────────────────────────────────────────────────

RETRAIN = not (MODEL_PKL.exists() and HISTORY_PKL.exists() and TESTDATA.exists())

if RETRAIN:
    print('Loading response matrices ...')
    Y_c = load_response(CONCRETE_DIR, Thickness_concrete)
    Y_s = load_response(STEEL_DIR,    Thickness_steel)
    Y_b = load_response(BPE_DIR,      Thickness_BPE)

    print('Generating training data ...')
    xc, yc = generate_output_flux(Y_c, Thickness_concrete, DENSITY['Concrete'])
    xs, ys = generate_output_flux(Y_s, Thickness_steel,    DENSITY['Steel'])
    xb, yb = generate_output_flux(Y_b, Thickness_BPE,      DENSITY['BPE'])

    x = np.concatenate([xc.reshape(-1, 250, 3),
                        xs.reshape(-1, 250, 3),
                        xb.reshape(-1, 250, 3)])
    y = np.concatenate([yc.reshape(-1, 250, 1),
                        ys.reshape(-1, 250, 1),
                        yb.reshape(-1, 250, 1)])
    print(f'Dataset: {x.shape[0]} samples  (train+val+test)')

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

    model = Sequential([
        Input(shape=(250, 3)),
        Conv1D(64, 100, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64,  25, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64,  11, activation='relu', padding='same'),
        Conv1D(64,   3, activation='relu', padding='same'),
        Flatten(),
        Dense(256),
        Dense(64),
        Dense(250),
    ], name=f'k100s2_{args.source}')
    model.compile(optimizer='adam', loss=args.loss, metrics=['mae', 'mse'])
    model.summary()

    print(f'\nTraining for {args.epochs} epochs ...')
    history = model.fit(xtrain, ytrain, epochs=args.epochs,
                        validation_split=0.3, verbose=2)
    hist = history.history

    with open(MODEL_PKL,   'wb') as f: pickle.dump(model, f)
    with open(HISTORY_PKL, 'wb') as f: pickle.dump(hist,  f)
    np.savez(TESTDATA, xtest=xtest, ytest=ytest)
    print(f'\nSaved → {SAVE_DIR}')

else:
    print(f'Model already exists in {SAVE_DIR} — skipping training.')
    with open(MODEL_PKL,   'rb') as f: model = pickle.load(f)
    with open(HISTORY_PKL, 'rb') as f: hist  = pickle.load(f)
    td = np.load(TESTDATA)
    xtest, ytest = td['xtest'], td['ytest']


# ── loss curve ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(hist['loss'],     label='train')
ax.plot(hist['val_loss'], label='val')
ax.set_xlabel('Epoch'); ax.set_ylabel(args.loss.upper())
ax.set_title(f'k100s2  —  source: {args.source}')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR / 'loss_curve.png')
plt.close()


# ── sample prediction plots ───────────────────────────────────────────────────

R = np.linspace(1, 250, 250)
y_pred = model.predict(xtest).flatten()
density_map = {2.3: 'Concrete', 7.86: 'Steel', 1.04: 'BPE'}

for i in range(20):
    pred = y_pred[250 * i : 250 * (i + 1)]
    t    = round(xtest[i, 0, 1] * 1000)
    d    = round(xtest[i, 0, 2] * 1000, 2)
    mat  = density_map[min(density_map, key=lambda k: abs(k - d))]
    plt.figure(dpi=150)
    plt.plot(R, pred,      'r--', label='ML predicted')
    plt.plot(R, ytest[i],  'k-',  label='Test output')
    plt.xlabel('Energy (MeV)'); plt.ylabel('log₁₀(Flux)')
    plt.title(f'{mat} {t} cm  —  source: {args.source}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f'{mat}_{t}cm_sample{i+1}.png')
    plt.close()

print('\nDone.')
