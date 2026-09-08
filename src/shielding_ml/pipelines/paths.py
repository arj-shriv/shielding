"""
Repository path resolution.

All paths are relative to REPO_ROOT, which is auto-detected as the directory
containing pyproject.toml (or set via the SHIELDING_ML_ROOT env variable).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    env = os.environ.get("SHIELDING_ML_ROOT")
    if env:
        return Path(env).resolve()
    # Walk up from this file looking for pyproject.toml
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Cannot locate repository root. "
        "Set SHIELDING_ML_ROOT environment variable or run from within the repo."
    )


REPO_ROOT: Path = _find_repo_root()

# ── Stable directory aliases ──────────────────────────────────────────────────

DATA_RAW       = REPO_ROOT / "data" / "raw"
DATA_REF       = REPO_ROOT / "data" / "reference"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

PHITS_HE       = DATA_RAW / "phits" / "high_energy"
PHITS_LE       = DATA_RAW / "phits" / "low_energy"
PHITS_ML       = DATA_RAW / "phits" / "multilayer"
PHITS_3L       = DATA_RAW / "phits" / "3layer"
RESPONSE_MATS  = DATA_RAW / "response_matrices"
SPECTRA_DIR    = DATA_RAW / "spectra"

MODELS_DIR     = REPO_ROOT / "models"
RUNS_DIR       = REPO_ROOT / "runs"
REPORTS_DIR    = REPO_ROOT / "reports"
REPORTS_SINGLE = REPORTS_DIR / "single_layer"
REPORTS_ML     = REPORTS_DIR / "multilayer"
CONFIGS_DIR    = REPO_ROOT / "configs"

# ── Reference file shortcuts ──────────────────────────────────────────────────

DOSE_TABLE     = DATA_REF / "Energy_to_Effective_Dose.txt"
TRACKNET10_SPECTRUM = SPECTRA_DIR / "TrackNet10_spectrum.dat"
