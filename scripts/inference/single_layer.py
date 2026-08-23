"""
Single-layer neutron shielding CNN inference.

Importable by the Streamlit app, the inference notebook, and standalone scripts.

Public API
----------
    load_model(path)                         -> keras.Model
    predict_flux(model, material, thickness) -> np.ndarray  shape (250,)
    compute_dose(flux)                       -> float        mrem/hr

Input tensor layout (shape [1, 250, 3]):
    channel 0 : source spectrum (normalised, sums to ~1)
    channel 1 : thickness  × 0.001   (e.g. 45 cm → 0.045)
    channel 2 : density    × 0.001   (e.g. 2.3 g/cm³ → 0.0023)

Output: log10(flux) for 250 bins → flux = 10^pred × FLUX_SCALE (2,000,000)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# ── package imports ───────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.constants import FLUX_SCALE, MATERIALS, E_ALL
from shielding_ml.data.loaders import load_spectrum
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.pipelines.paths import (
    REPO_ROOT, TRACKNET10_SPECTRUM, DOSE_TABLE, DATA_REF,
)

# ── module-level constants ────────────────────────────────────────────────────
INPUT_SCALE = 0.001          # scales thickness (cm) and density (g/cm³) to ~[0,1]
N_BINS      = 250            # energy bins: 0.5, 1.5, …, 249.5 MeV

# Shared resources loaded once at import time
_A_SPEC : np.ndarray | None = None   # TrackNet10 source spectrum
_R_ALL  : np.ndarray | None = None   # ICRP dose coefficients


def _get_shared_resources() -> tuple[np.ndarray, np.ndarray]:
    """Lazy-load shared spectrum and dose table (cached after first call)."""
    global _A_SPEC, _R_ALL
    if _A_SPEC is None:
        _A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
    if _R_ALL is None:
        _R_ALL = _build_r_all(DOSE_TABLE)
    return _A_SPEC, _R_ALL


# ── public API ────────────────────────────────────────────────────────────────

def load_model(path: str | Path):
    """Load a Keras model from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_flux(
    model,
    material:     str,
    thickness_cm: float,
    source_spec:  np.ndarray | None = None,
) -> np.ndarray:
    """
    Run one forward pass and return the flux spectrum.

    Parameters
    ----------
    model        : loaded Keras model (from load_model)
    material     : 'Concrete', 'Steel', or 'BPE'
    thickness_cm : shield thickness in cm
    source_spec  : 250-element source spectrum; defaults to TrackNet10

    Returns
    -------
    flux : np.ndarray shape (250,) in n/cm²/source-neutron
    """
    if source_spec is None:
        source_spec, _ = _get_shared_resources()

    density = MATERIALS[material]

    thickness_ch = np.full(N_BINS, thickness_cm * INPUT_SCALE)
    density_ch   = np.full(N_BINS, density      * INPUT_SCALE)

    X = np.stack([source_spec, thickness_ch, density_ch], axis=1)  # (250, 3)
    X = X.reshape(1, N_BINS, 3)

    log_flux = model.predict(X, verbose=0).flatten()
    return 10 ** log_flux * FLUX_SCALE


def compute_dose(flux: np.ndarray) -> float:
    """
    Convert a flux spectrum to effective dose rate (mrem/hr).

    Uses ICRP fluence-to-effective-dose coefficients, midpoint integration,
    excluding bin 0 (0–1 MeV thermal region).
    """
    _, R_ALL = _get_shared_resources()
    return dose_midpoint(flux, R_ALL, excl_bin0=True)


def get_void_flux() -> np.ndarray:
    """Return the unshielded reference flux from No_Shielding.out."""
    from shielding_ml.data.loaders import load_phits
    flux, _ = load_phits(str(DATA_REF / "No_Shielding.out"))
    return flux
