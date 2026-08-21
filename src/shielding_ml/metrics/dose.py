"""
Dose integration strategies.

Two implementations are preserved deliberately:
  - midpoint_r:   R evaluated at bin centres (0.5, 1.5, … 249.5 MeV)
  - geometric_r:  R = sqrt(R(edge_lo) * R(edge_hi)); bin-0 lower edge = 1 keV

Both exclude bin-0 (0–1 MeV) by default when excl_bin0=True.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

from shielding_ml.data.constants import E_ALL


def _build_r_all(dose_table_path: str | Path) -> np.ndarray:
    """Load ICRP table and interpolate R at E_ALL midpoints."""
    E_tbl, R_tbl = np.loadtxt(dose_table_path, unpack=True)
    return interp1d(E_tbl, R_tbl)(E_ALL)


def _build_r_geom(dose_table_path: str | Path) -> np.ndarray:
    """
    Build geometric-mean response array.
    Bin i has edges [i MeV, (i+1) MeV]; bin 0 lower edge = 1e-3 MeV (1 keV).
    R_geom[i] = sqrt(R(lo) * R(hi))
    """
    E_tbl, R_tbl = np.loadtxt(dose_table_path, unpack=True)
    f = interp1d(E_tbl, R_tbl, bounds_error=False,
                 fill_value=(R_tbl[0], R_tbl[-1]))
    lo = np.arange(250, dtype=float)
    lo[0] = 1e-3
    hi = lo + 1.0
    hi[0] = 1.0
    return np.sqrt(f(lo) * f(hi))


def dose_midpoint(
    flux: np.ndarray,
    r_all: np.ndarray,
    excl_bin0: bool = True,
    max_bin: int = 250,
) -> float:
    """
    Dose using R sampled at bin midpoints.

    Parameters
    ----------
    flux    : shape (250,)  neutron flux per bin
    r_all   : shape (250,)  ICRP response at E_ALL midpoints
    excl_bin0 : exclude bin-0 (0–1 MeV)
    max_bin   : integrate up to this bin index (exclusive); default 250 = full range
    """
    start = 1 if excl_bin0 else 0
    return float(np.sum(flux[start:max_bin] * r_all[start:max_bin]))


def dose_geometric(
    flux: np.ndarray,
    r_geom: np.ndarray,
    excl_bin0: bool = True,
    max_bin: int = 250,
) -> float:
    """
    Dose using geometric-mean R at bin edges (see module docstring).

    Parameters
    ----------
    flux   : shape (250,)
    r_geom : shape (250,)  from _build_r_geom()
    excl_bin0 : exclude bin-0
    max_bin   : integrate up to this bin index (exclusive)
    """
    start = 1 if excl_bin0 else 0
    return float(np.sum(flux[start:max_bin] * r_geom[start:max_bin]))


def integrate_le_dose(
    e_lo: np.ndarray,
    e_hi: np.ndarray,
    flux: np.ndarray,
    dose_table_path: str | Path,
) -> float:
    """
    Integrate dose over LE bins using trapezoid rule.
    ΔE = e_hi − e_lo (variable, log-spaced bins).
    """
    E_tbl, R_tbl = np.loadtxt(dose_table_path, unpack=True)
    f = interp1d(E_tbl, R_tbl)
    e_mid = (e_lo + e_hi) / 2.0
    dE = e_hi - e_lo
    R = f(np.clip(e_mid, E_tbl[0], E_tbl[-1]))
    return float(np.sum(flux * R * dE))
