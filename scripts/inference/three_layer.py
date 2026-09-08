"""
Three-layer (triple-chained) neutron shielding inference.

Extends the two-layer chaining logic in scripts/inference/multilayer.py by one
more hop. Layer 1 always runs k100s2_v1 with the TrackNet10 source (as in every
multilayer variant); layers 2 and 3 — the "deeper layers" — always run
transfer_mlp, since that's the model actually trained to correct predictions
fed a layer-2+ style source.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHERE THE AMPLITUDE CORRECTION (a1 × a2) COMES FROM

For an N-layer chain, exactly N-1 amplitude factors (a_1 .. a_{N-1}) always end
up applied to the final layer's raw output, regardless of what source each
intermediate layer is fed:

  - If layer i+1 is fed TrackNet10 (ignoring layer i's output entirely), layer
    i's own transmission fraction a_i never reaches layer i+1's forward pass
    at all — it has to be reinstated externally.
  - If layer i+1 is fed layer i's ACTUAL output, that output is renormalised
    to sum=1 first (the model's input contract expects a unit-sum source) —
    which discards layer i's absolute scale (a_i) just as thoroughly. Only
    the *shape* survives the renormalisation.

Either way, a_i is lost and has to be multiplied back in once, at the end.
This is the same rule scripts/inference/multilayer.py's two functions already
follow (predict_chained_cnn_source and predict_tracknet_source both apply
exactly one factor, a_1, for their 2-layer chains — there's no a_2 for a
2-layer chain because there's no layer 3 downstream to have "lost" it to).
The last layer's own attenuation is never separately corrected — it's already
correctly baked into that layer's own raw output, whatever shape it was given.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four source-chaining combinations for layers 2 and 3 (each ∈ {tracknet10, actual}):

  APPROACH 1  l2_source='tracknet10'  l3_source='tracknet10'
      Amplitude-only in both extra layers — every layer sees a full-intensity
      TrackNet10 beam in isolation; a1 and a2 (both computed against void)
      carry all of the cross-layer information.

  APPROACH 2  l2_source='tracknet10'  l3_source='actual'
      Layer 2 sees TrackNet10 (in-distribution); layer 3 sees layer 2's own
      predicted output (renormalised) as its source.

  APPROACH 3  l2_source='actual'      l3_source='tracknet10'
      Layer 2 sees layer 1's actual output (renormalised); layer 3 reverts to
      a fresh TrackNet10 beam, ignoring layer 2's predicted shape.

  APPROACH 4  l2_source='actual'      l3_source='actual'
      Full shape chaining — each layer sees the previous layer's actual
      (renormalised) predicted output. Most physically motivated, but also
      the most exposed to out-of-distribution inputs (the failure mode
      documented for the 2-layer analogue, predict_chained_cnn_source).

Usage:
    from scripts.inference.three_layer import predict_3layer

    flux = predict_3layer(
        model_v1, model_transfer,
        mat1='Steel', thick1=40, mat2='Concrete', thick2=85, mat3='BPE', thick3=20,
        l2_source='tracknet10', l3_source='actual',   # APPROACH 2
    )
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.inference.single_layer import predict_flux, get_void_flux, _get_shared_resources
from scripts.inference.multilayer   import _amplitude_factor, _BIN_SLICE

_VALID_SOURCES = ("tracknet10", "actual")

APPROACH_LABELS = {
    (1, "tracknet10", "tracknet10"): "1 — amplitude-only (TrackNet10 in both extra layers)",
    (2, "tracknet10", "actual"):     "2 — TrackNet10 → L2, actual L2 output → L3",
    (3, "actual", "tracknet10"):     "3 — actual L1 output → L2, TrackNet10 → L3",
    (4, "actual", "actual"):         "4 — actual output chained through both extra layers",
}


def _resolve_source(mode: str, upstream_raw: np.ndarray, a_spec: np.ndarray) -> np.ndarray:
    if mode == "tracknet10":
        return a_spec
    if mode == "actual":
        return upstream_raw / upstream_raw.sum()
    raise ValueError(f"source mode must be one of {_VALID_SOURCES}, got {mode!r}")


def predict_3layer(
    model_v1,
    model_transfer,
    mat1: str, thick1: float,
    mat2: str, thick2: float,
    mat3: str, thick3: float,
    l2_source: str = "tracknet10",
    l3_source: str = "tracknet10",
) -> np.ndarray:
    """
    Three-layer chained prediction. See module docstring for the derivation of
    the a1 × a2 amplitude correction and what each (l2_source, l3_source)
    combination means physically.

    Layer 1 always: k100s2_v1, TrackNet10 source (gives a1).
    Layers 2 and 3 always: model_transfer ("k100 + MLP"), fed either a fresh
    TrackNet10 beam or the previous layer's actual (renormalised) output,
    per l2_source / l3_source.
    """
    if l2_source not in _VALID_SOURCES or l3_source not in _VALID_SOURCES:
        raise ValueError(f"l2_source and l3_source must be one of {_VALID_SOURCES}")

    A_SPEC, _ = _get_shared_resources()
    void_flux = get_void_flux()

    # Layer 1 — always k100s2_v1 with TrackNet10, purely for a1.
    a1, flux_L1 = _amplitude_factor(model_v1, mat1, thick1)

    # Layer 2 — always transfer_mlp.
    source_L2   = _resolve_source(l2_source, flux_L1, A_SPEC)
    flux_L2_raw = predict_flux(model_transfer, mat2, thick2, source_spec=source_L2)
    a2 = float(flux_L2_raw[_BIN_SLICE].sum() / void_flux[_BIN_SLICE].sum())

    # Layer 3 — always transfer_mlp.
    source_L3   = _resolve_source(l3_source, flux_L2_raw, A_SPEC)
    flux_L3_raw = predict_flux(model_transfer, mat3, thick3, source_spec=source_L3)

    return flux_L3_raw * a1 * a2


def predict_3layer_all_approaches(
    model_v1, model_transfer,
    mat1: str, thick1: float,
    mat2: str, thick2: float,
    mat3: str, thick3: float,
) -> dict[str, np.ndarray]:
    """Run all four approaches, keyed by their APPROACH_LABELS description."""
    results = {}
    for (n, l2s, l3s), label in APPROACH_LABELS.items():
        results[label] = predict_3layer(
            model_v1, model_transfer,
            mat1, thick1, mat2, thick2, mat3, thick3,
            l2_source=l2s, l3_source=l3s,
        )
    return results
