"""
Train a transfer head on top of the frozen k100s2_v1 backbone, and register
the result in models/registry.json.

Training data (all three --arch options): run the frozen v1 model on
Concrete/Steel/BPE at 5 thicknesses each (TrackNet10 input, as layer 1 always
uses), normalise each output → use as the layer-2 "source" for training
samples; sample a random layer-2 (material, thickness) and compute the target
flux via the response matrix. This matches what a transfer head actually sees
at multilayer inference time — it's the approach that produced the primary
transfer_mlp model (see train_transfer_cnn_source.py, the legacy script this
consolidates).

--arch mlp     Dense(64, relu) → Dense(250)   [PRIMARY architecture, 32,314 new params]
--arch linear  Dense(250)                     [63,250 new params, historically underpowered]
--arch conv    Conv1D(50,k50) → Conv1D(10,k10,s2) → Flatten → Dense(250)  [185,310 new params]
               Cuts the backbone after its last Conv1D layer instead of at the final output.

Usage
-----
    ../venv/bin/python scripts/training/train_transfer.py --name transfer_mlp_v2 --arch mlp
    ../venv/bin/python scripts/training/train_transfer.py --name my_conv_head --arch conv --epochs 25
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

from shielding_ml.data.loaders import load_spectrum
from shielding_ml.pipelines.paths import RESPONSE_MATS, MODELS_DIR, TRACKNET10_SPECTRUM, REPO_ROOT
from shielding_ml.data.constants import FLUX_SCALE

from scripts.training.registry_utils import (
    require_available, upsert_entry, write_description, load_registry, base_model_pkl,
)

K100S2_BASE = MODELS_DIR / "k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240"
DENSITY = {"Concrete": 2.3, "Steel": 7.86, "BPE": 1.04}

SRC_THICKNESSES = {
    "Concrete": [25, 50, 75, 100, 125],
    "Steel":    [20, 40, 60, 80, 100],
    "BPE":      [20, 40, 60, 80, 100],
}
L2_THICKNESSES = {
    "Concrete": list(range(5,  155, 5)),
    "Steel":    list(range(10, 110, 10)),
    "BPE":      list(range(10, 110, 10)),
}
L2_MATERIALS = list(L2_THICKNESSES.keys())

ARCH_DESCRIPTIONS = {
    "mlp":    "Dense(64, ReLU) → Dense(250)",
    "linear": "Dense(250)",
    "conv":   "Conv1D(50,k50,s1,same) → Conv1D(10,k10,s2,same) → Flatten → Dense(250)",
}


def load_response(material: str, thickness_cm: int, cache: dict) -> np.ndarray:
    key = (material, thickness_cm)
    if key not in cache:
        flux = np.loadtxt(RESPONSE_MATS / material / f"{thickness_cm}cm"
                           / "Neutron_Diff_Flux_Response.dat")
        R = flux.T.reshape(250, 250)
        cache[key] = R.T / (200 * 200 * 50)
    return cache[key]


def build_head(base_model, arch: str, input_scale: float):
    """Return (keras.Model transfer_model, frozen_output_layer_shape)."""
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Conv1D, Flatten

    inp = Input(shape=(250, 3), name="input")

    if arch in ("mlp", "linear"):
        base_model.trainable = False
        base_out = base_model(inp, training=False)   # (batch, 250) frozen
        if arch == "mlp":
            x = Dense(64, activation="relu", name="head_hidden")(base_out)
            out = Dense(250, name="correction_head")(x)
        else:
            out = Dense(250, name="correction_head")(base_out)
        return Model(inputs=inp, outputs=out, name=f"k100s2_transfer_{arch}")

    # arch == "conv": cut the backbone after its last Conv1D/pooling layer,
    # not at the final Dense output.
    backbone_layers = []
    for layer in base_model.layers:
        lname = layer.name.lower()
        if "conv1d" in lname or "max_pool" in lname or "pooling" in lname:
            backbone_layers.append(layer)
        elif backbone_layers:
            break
    x = inp
    for layer in backbone_layers:
        layer.trainable = False
        x = layer(x, training=False)
    x = Conv1D(50, kernel_size=50, strides=1, padding="same",
               activation="relu", name="conv_head_0")(x)
    x = Conv1D(10, kernel_size=10, strides=2, padding="same", name="conv_head_1")(x)
    x = Flatten(name="head_flatten")(x)
    out = Dense(250, name="head_output")(x)
    return Model(inputs=inp, outputs=out, name="k100s2_transfer_conv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",        required=True,
                   help="Model name to register (must be unique unless --overwrite).")
    p.add_argument("--arch",        default="mlp", choices=["mlp", "linear", "conv"])
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--loss",        default="mae", choices=["mae", "mse"])
    p.add_argument("--input_scale", type=float, default=0.001)
    p.add_argument("--n_per_src",   type=int,   default=10_000,
                   help="Training samples per source material (spread across 5 thicknesses).")
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
    print(f"\n=== transfer head  name={args.name}  arch={args.arch}  "
          f"epochs={args.epochs}  section={section} ===")
    print(f"Save dir: {save_dir}\n")

    registry = load_registry()
    with open(base_model_pkl(registry), "rb") as f:
        base_model = pickle.load(f)
    print(f"Base model loaded: {registry['primary']['k100s2_v1']['pkl']}")

    transfer_model = build_head(base_model, args.arch, args.input_scale)
    transfer_model.compile(optimizer="adam", loss=args.loss, metrics=["mae", "mse"])
    transfer_model.summary()
    trainable = sum(int(np.prod(w.shape)) for w in transfer_model.trainable_weights)
    print(f"\nNew trainable params: {trainable:,}\n")

    # ── generate CNN layer-1 source spectra ───────────────────────────────────
    A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
    print("Generating CNN layer-1 source spectra ...")
    src_spectra: list[tuple[str, int, np.ndarray]] = []
    for mat, thicks in SRC_THICKNESSES.items():
        density = DENSITY[mat]
        for thick in thicks:
            B  = np.ones(250) * thick   * args.input_scale
            B1 = np.ones(250) * density * args.input_scale
            X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
            log_pred = base_model.predict(X, verbose=0).flatten()
            pred = 10 ** log_pred * FLUX_SCALE
            s = pred.sum()
            norm = pred / s if s > 0 else pred
            src_spectra.append((mat, thick, norm))
            print(f"  {mat} {thick}cm  sum={s:.3e}")

    # ── generate training data ────────────────────────────────────────────────
    n_per_spec = args.n_per_src // 5
    print(f"\nGenerating {n_per_spec} samples per source spectrum "
          f"({len(src_spectra)} spectra) ...")

    response_cache: dict = {}
    all_x, all_y = [], []
    rng = np.random.default_rng(42)
    for src_mat, src_thick, src_norm in src_spectra:
        for _ in range(n_per_spec):
            l2_mat   = rng.choice(L2_MATERIALS)
            l2_thick = int(rng.choice(L2_THICKNESSES[l2_mat]))
            l2_dens  = DENSITY[l2_mat]
            R = load_response(l2_mat, l2_thick, response_cache)
            target = R @ src_norm
            if target.sum() == 0:
                continue
            B  = np.ones(250) * l2_thick * args.input_scale
            B1 = np.ones(250) * l2_dens  * args.input_scale
            all_x.append(np.array([src_norm, B, B1]).T)
            all_y.append(np.where(target > 0, np.log10(target), 0.0))

    x = np.array(all_x)
    y = np.array(all_y)
    print(f"Dataset: {x.shape[0]} samples\n")

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"Training for {args.epochs} epochs ...")
    history = transfer_model.fit(xtrain, ytrain, epochs=args.epochs,
                                  validation_split=0.2, batch_size=64, verbose=2)
    hist = history.history

    model_pkl = save_dir / "transfer_model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(transfer_model, f)
    with open(save_dir / "history.pkl", "wb") as f:
        pickle.dump(hist, f)
    np.savez(save_dir / "testdata.npz", xtest=xtest, ytest=ytest)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(hist["loss"],     label="train")
    ax.plot(hist["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(args.loss.upper())
    ax.set_title(f"Transfer head — {args.name}  (arch: {args.arch})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png")
    plt.close(fig)

    y_pred = transfer_model.predict(xtest, verbose=0)
    val_mae = float(np.mean(np.abs(y_pred - ytest)))
    print(f"\nTest-set MAE: {val_mae:.4f}")

    description = args.description or (
        f"Transfer head ({args.arch}: {ARCH_DESCRIPTIONS[args.arch]}) on the frozen "
        f"k100s2_v1 backbone. Trained on actual v1 layer-1 CNN predictions as layer-2 "
        f"source spectra ({args.epochs} epochs, {args.loss.upper()} loss). "
        f"Registered via train_transfer.py."
    )

    entry = {
        "pkl":              str(model_pkl.relative_to(REPO_ROOT)),
        "architecture":     f"Frozen k100s2_v1 backbone → {ARCH_DESCRIPTIONS[args.arch]}",
        "params":           trainable,
        "loss":             args.loss.upper(),
        "input_scale":      args.input_scale,
        "training_source":  "CNN layer-1 predictions (Concrete/Steel/BPE, 5 thicknesses each)",
        "val_mae":          round(val_mae, 4),
        "created":          date.today().isoformat(),
        "status":           "active" if args.primary else "experimental",
        "use_for":          ["multilayer_layer2"],
        "description":      description,
    }
    upsert_entry(args.name, section, entry)
    write_description(save_dir, description)

    print(f"\nRegistered '{args.name}' in models/registry.json [{section}]")
    print(f"Saved → {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
