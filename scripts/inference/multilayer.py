"""
Multilayer (chained) neutron shielding inference — all variants.

Each public function is one self-contained chaining strategy. They are named
and documented so the approach is unambiguous to a reader.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIANT A  chained_cnn_source   (baseline, not recommended)
   Layer-2 source = normalised CNN layer-1 output flux.
   Result: avg_ratio 3.3–6.4× for concrete-first, 0.31–0.40× for steel-first.
   Why it fails: the normalised CNN output is out-of-distribution for the
   model trained on smooth TrackNet10 spectra.

VARIANT B  tracknet_source      *** PRIMARY / RECOMMENDED ***
   Layer-2 source = original TrackNet10 beam (keeps layer-2 in-distribution).
   Layer-1 amplitude attenuation (a-factor) extracted from k100s2_v1 and
   applied to the final layer-2 output to restore absolute scale.
   Result: dose error −28% to +18% across all 6 multilayer cases.

VARIANT C  bimodel
   Layer 2 runs a separate k100s2 trained on a shielded source spectrum
   (e.g. concrete25 piecewise fit) to better match the layer-1 output shape.
   Result: negligible improvement over Variant A — the amplitude error
   from the a-factor dominates regardless of the layer-2 model.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage example (Variant B — primary mode):
    from scripts.inference.multilayer import predict_tracknet_source

    flux = predict_tracknet_source(
        model_v1       = load_model('.../primary/k100s2_v1/model.pkl'),
        model_transfer = load_model('.../primary/transfer_mlp/transfer_model.pkl'),
        mat1='Concrete', thick1=45,
        mat2='Steel',    thick2=33,
    )
    dose = compute_dose(flux)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from scripts.inference.single_layer import (
    predict_flux, compute_dose, get_void_flux, _get_shared_resources,
)

# Energy window used for amplitude factor: 20–150 MeV (bins 20–149)
_BIN_SLICE = slice(20, 150)


# ── shared helper ─────────────────────────────────────────────────────────────

def _amplitude_factor(model_v1, mat1: str, thick1: float) -> tuple[float, np.ndarray]:
    """
    Run layer-1 CNN and compute the amplitude attenuation factor.

    a = sum(CNN_L1_flux[20:150]) / sum(void_flux[20:150])

    This ratio captures how much the first shield layer attenuates the beam
    in the 20–150 MeV window. It is applied to the layer-2 output so the
    final flux has the correct absolute scale, even though the layer-2 source
    was fed a normalised (unit-sum) spectrum.

    Returns (a_factor, layer1_flux).
    """
    A_SPEC, _ = _get_shared_resources()
    flux_L1   = predict_flux(model_v1, mat1, thick1, source_spec=A_SPEC)
    void_flux = get_void_flux()
    a = flux_L1[_BIN_SLICE].sum() / void_flux[_BIN_SLICE].sum()
    return float(a), flux_L1


# ── VARIANT A — baseline chaining (not recommended) ──────────────────────────

def predict_chained_cnn_source(
    model_v1,
    mat1: str, thick1: float,
    mat2: str, thick2: float,
) -> np.ndarray:
    """
    VARIANT A — baseline chaining.

    Layer 1: k100s2_v1 with TrackNet10 source.
    Layer 2: same model with *normalised CNN layer-1 output* as source.
    Amplitude: a-factor from layer-1 flux applied to layer-2 output.

    Known limitation: the normalised CNN output is spectrally different from
    the smooth TrackNet10 distributions the model was trained on, causing
    systematic amplitude errors (3–6× overestimate for concrete-first stacks).
    """
    A_SPEC, _ = _get_shared_resources()

    # Layer 1
    a_factor, flux_L1 = _amplitude_factor(model_v1, mat1, thick1)

    # Normalise layer-1 flux to use as layer-2 source
    source_L2 = flux_L1 / flux_L1.sum()

    # Layer 2 — model sees the normalised CNN output as its source
    flux_L2_raw = predict_flux(model_v1, mat2, thick2, source_spec=source_L2)

    return flux_L2_raw * a_factor


# ── VARIANT B — TrackNet10 source (PRIMARY) ───────────────────────────────────

def predict_tracknet_source(
    model_v1,
    model_transfer,
    mat1: str, thick1: float,
    mat2: str, thick2: float,
) -> np.ndarray:
    """
    VARIANT B — TrackNet10 source for layer 2.  *** PRIMARY / RECOMMENDED ***

    Layer 1: k100s2_v1 with TrackNet10 source → extracts a-factor only.
    Layer 2: transfer_mlp with the *original TrackNet10 source* (not the CNN
             layer-1 output). This keeps the layer-2 model in-distribution.
    Final flux: layer-2 output × a-factor.

    The a-factor carries the layer-1 amplitude information so the result
    reflects the full two-layer attenuation, even though layer-2 was fed
    the unshielded TrackNet10 beam.

    Performance: dose error −28% to +18% across all 6 multilayer PHITS cases.
    """
    A_SPEC, _ = _get_shared_resources()

    # Layer 1: run to get amplitude attenuation only
    a_factor, _ = _amplitude_factor(model_v1, mat1, thick1)

    # Layer 2: feed original TrackNet10 source (in-distribution)
    flux_L2_raw = predict_flux(model_transfer, mat2, thick2, source_spec=A_SPEC)

    return flux_L2_raw * a_factor


# ── VARIANT C — bi-model (source-matched layer-2) ─────────────────────────────

def predict_bimodel(
    model_v1,
    model_l2,
    mat1: str, thick1: float,
    mat2: str, thick2: float,
    source_spec_l2: np.ndarray | None = None,
) -> np.ndarray:
    """
    VARIANT C — bi-model.

    Layer 1: k100s2_v1 with TrackNet10 source → a-factor.
    Layer 2: a *separate* k100s2 variant trained on a shielded source spectrum
             (e.g. v_concrete25, v_steel27) so its input distribution is closer
             to a real shielded beam.

    Source for layer 2 defaults to TrackNet10; pass a piecewise shielded
    spectrum as source_spec_l2 to test the matched-source case.

    Historical result: avg_ratio nearly identical to Variant A. No measurable
    improvement — the amplitude error dominates regardless of model choice.
    """
    A_SPEC, _ = _get_shared_resources()
    src_l2 = source_spec_l2 if source_spec_l2 is not None else A_SPEC

    a_factor, _ = _amplitude_factor(model_v1, mat1, thick1)
    flux_L2_raw = predict_flux(model_l2, mat2, thick2, source_spec=src_l2)

    return flux_L2_raw * a_factor
