"""
Model loading and inference utilities.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from shielding_ml.data.constants import FLUX_SCALE, MATERIALS


def load_model(pkl_path: str | Path):
    """Load a serialized Keras model from a pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def build_input(
    spectrum: np.ndarray,
    thickness: float,
    density: float,
    scale: float,
) -> np.ndarray:
    """
    Build the (1, 250, 3) input array for the CNN.

    Channels: [normalized_spectrum, thickness*scale, density*scale]
    """
    B  = np.ones(250) * thickness * scale
    B1 = np.ones(250) * density   * scale
    return np.array([spectrum, B, B1]).T.reshape(1, 250, 3)


def predict_flux(
    model,
    spectrum: np.ndarray,
    material: str,
    thickness: float,
    scale: float,
) -> np.ndarray:
    """
    Run CNN inference and return flux in physical units.

    Returns shape (250,): 10^(model_output) * FLUX_SCALE
    """
    density = MATERIALS[material]
    X = build_input(spectrum, thickness, density, scale)
    log_pred = model.predict(X, verbose=0).flatten()
    return (10.0 ** log_pred) * FLUX_SCALE
