"""
Train the k100s2 architecture with a selectable source spectrum, and register
the result in models/registry.json so both Streamlit apps can see it.

Architecture: Conv1D(64,100,s2)→MaxPool→Conv1D(64,25,s2)→MaxPool
              →Conv1D(64,11)→Conv1D(64,3)→Flatten→Dense(256)→Dense(64)→Dense(250)

Usage
-----
    ../venv/bin/python scripts/training/train_k100s2.py --name v_concrete25 --source concrete25
    ../venv/bin/python scripts/training/train_k100s2.py --name my_new_model --source steel27 --epochs 25
    ../venv/bin/python scripts/training/train_k100s2.py --name k100s2_v2 --source tracknet10 --primary

If --name already exists in the registry, training is refused unless
--overwrite is passed — this is the "exists check" from the package proposal.
New models land in models/<...>/experimental/<name>/ by default; --primary
puts them in .../primary/<name>/ instead.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shielding_ml.data.source_spectra import SOURCE_FUNCTIONS
from shielding_ml.pipelines.paths import RESPONSE_MATS, MODELS_DIR, REPO_ROOT

from scripts.training.registry_utils import (
    require_available, upsert_entry, write_description,
)

K100S2_BASE = MODELS_DIR / "k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240"
DENSITY = {"Concrete": 2.3, "Steel": 7.86, "BPE": 1.04}

THICKNESSES = {
    "Concrete": np.array([5.0] + list(np.linspace(10, 150, 15))),  # 16 values
    "Steel":    np.linspace(10, 100, 10),
    "BPE":      np.linspace(10, 100, 10),
}


def load_response(material: str, thicknesses: np.ndarray) -> np.ndarray:
    mat_dir = RESPONSE_MATS / material
    Y1 = []
    for t in thicknesses:
        flux = np.loadtxt(mat_dir / f"{int(t)}cm" / "Neutron_Diff_Flux_Response.dat")
        flux = flux.T
        for j in range(250):
            Y1.append(list(flux[j]))
    Y1 = np.array(Y1).reshape(len(thicknesses), 250, 250)
    Y2 = np.array([Y1[i].T for i in range(len(thicknesses))]).T
    return Y2 / (200 * 200 * 50)


def generate_output_flux(Y, thicknesses, density, source_fn, input_scale, n_bins_min, n_bins_max):
    E_MeV   = np.linspace(0.5, 249.5, 250)
    weights = source_fn(E_MeV)
    X_new, Y_new = [], []
    for k in range(len(thicknesses)):
        for _ in range(10_000):
            A   = np.zeros(250)
            w1  = np.random.randint(n_bins_min, n_bins_max + 1)
            idx = np.random.randint(0, 249, size=w1)
            for j in idx:
                A[j] = weights[j]
            A[A == 0] = 1e-7
            A /= A.sum()
            B  = np.ones(250) * thicknesses[k] * input_scale
            B1 = np.ones(250) * density         * input_scale
            X_new.append(np.array([A, B, B1]).T)
            S = sum(Y[j, :, k] * A[j] for j in range(250))
            Y_new.append(np.where(S != 0, np.log10(S), 0.0))
    return np.array(X_new), np.array(Y_new)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",        required=True,
                   help="Model name to register (must be unique unless --overwrite).")
    p.add_argument("--source",      required=True, choices=list(SOURCE_FUNCTIONS.keys()))
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--loss",        default="mae", choices=["mae", "mse"])
    p.add_argument("--input_scale", type=float, default=0.001)
    p.add_argument("--n_bins_min",  type=int,   default=50)
    p.add_argument("--n_bins_max",  type=int,   default=240)
    p.add_argument("--primary",     action="store_true",
                   help="Save under primary/ instead of experimental/.")
    p.add_argument("--overwrite",   action="store_true",
                   help="Allow retraining a model name that's already registered.")
    p.add_argument("--description", default=None,
                   help="Override the auto-generated description.txt content.")
    args = p.parse_args()

    section = "primary" if args.primary else "experimental"

    blocked_section = require_available(args.name, args.overwrite)
    if blocked_section is not None:
        print(f"Model '{args.name}' already exists in registry section "
              f"'{blocked_section}'. Use --overwrite to replace it.")
        return 1

    save_dir = K100S2_BASE / section / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== k100s2  name={args.name}  source={args.source}  "
          f"epochs={args.epochs}  section={section} ===")
    print(f"Save dir: {save_dir}\n")

    print("Loading response matrices ...")
    Y = {mat: load_response(mat, thicks) for mat, thicks in THICKNESSES.items()}

    print("Generating training data ...")
    source_fn = SOURCE_FUNCTIONS[args.source]
    xs, ys = [], []
    for mat, thicks in THICKNESSES.items():
        x, y = generate_output_flux(Y[mat], thicks, DENSITY[mat], source_fn,
                                     args.input_scale, args.n_bins_min, args.n_bins_max)
        xs.append(x.reshape(-1, 250, 3))
        ys.append(y.reshape(-1, 250, 1))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    print(f"Dataset: {x.shape[0]} samples")

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

    model = Sequential([
        Input(shape=(250, 3)),
        Conv1D(64, 100, strides=2, activation="relu", padding="same", name="Conv1D_0"),
        MaxPooling1D(2, padding="same"),
        Conv1D(64,  25, strides=2, activation="relu", padding="same", name="Conv1D_1"),
        MaxPooling1D(2, padding="same"),
        Conv1D(64,  11, activation="relu", padding="same", name="Conv1D_2"),
        Conv1D(64,   3, activation="relu", padding="same", name="Conv1D_3"),
        Flatten(),
        Dense(256, name="Dense_1"),
        Dense(64,  name="Dense_2"),
        Dense(250),
    ], name=f"k100s2_{args.name}")
    model.compile(optimizer="adam", loss=args.loss, metrics=["mae", "mse"])
    model.summary()

    print(f"\nTraining for {args.epochs} epochs ...")
    history = model.fit(xtrain, ytrain, epochs=args.epochs, validation_split=0.3, verbose=2)
    hist = history.history

    model_pkl = save_dir / "model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(model, f)
    with open(save_dir / "history.pkl", "wb") as f:
        pickle.dump(hist, f)
    np.savez(save_dir / "testdata.npz", xtest=xtest, ytest=ytest)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(hist["loss"],     label="train")
    ax.plot(hist["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(args.loss.upper())
    ax.set_title(f"k100s2 — {args.name}  (source: {args.source})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close(fig)

    y_pred = model.predict(xtest, verbose=0)
    val_mae = float(np.mean(np.abs(y_pred.reshape(ytest.shape) - ytest)))
    print(f"\nTest-set MAE: {val_mae:.4f}")

    description = args.description or (
        f"k100s2 variant trained on the '{args.source}' source spectrum "
        f"({args.epochs} epochs, {args.loss.upper()} loss, bins "
        f"{args.n_bins_min}-{args.n_bins_max}). Registered via train_k100s2.py."
    )

    entry = {
        "pkl":              str(model_pkl.relative_to(REPO_ROOT)),
        "architecture":     "Conv1D k100(s2) → Conv1D k25(s2) → Conv1D k11 → Conv1D k3 "
                             "→ Dense 256 → Dense 64 → Dense 250",
        "params":           int(model.count_params()),
        "loss":             args.loss.upper(),
        "input_scale":      args.input_scale,
        "training_source":  args.source,
        "bin_range":        [args.n_bins_min, args.n_bins_max],
        "val_mae":          round(val_mae, 4),
        "created":          date.today().isoformat(),
        "status":           "active" if args.primary else "experimental",
        "use_for":          ["single_layer"] + (["multilayer_layer1"] if args.primary else []),
        "description":      description,
    }
    upsert_entry(args.name, section, entry)
    write_description(save_dir, description)

    print(f"\nRegistered '{args.name}' in models/registry.json [{section}]")
    print(f"Saved → {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
