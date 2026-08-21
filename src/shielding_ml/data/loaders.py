"""
PHITS T-Track file loaders and spectrum utilities.

All functions are pure (no side effects, no hardcoded paths).
Callers are responsible for resolving paths.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


def load_phits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a PHITS T-Track high-energy output file (250 uniform 1 MeV bins, 0–250 MeV).

    Returns
    -------
    flux : ndarray, shape (250,)
        Neutron flux per bin [n/cm²/source].
    rerr : ndarray, shape (250,)
        Relative statistical error per bin (NaN where not reported).
    """
    lines = Path(path).read_text().splitlines()
    hdr = next((ln for ln in lines if 'e-lower' in ln and 'neutron' in ln), None)
    if hdr is None:
        raise ValueError(f"No T-Track header found in {path}")
    idx = lines.index(hdr)
    flux, rerr = [], []
    for ln in lines[idx + 1 : idx + 251]:
        p = ln.split()
        if len(p) >= 4:
            flux.append(float(p[2])); rerr.append(float(p[3]))
        elif len(p) >= 3:
            flux.append(float(p[2])); rerr.append(float('nan'))
    return np.array(flux), np.array(rerr)


def load_phits_flux(path: str | Path) -> np.ndarray:
    """
    Parse a PHITS T-Track file and return only the flux array.
    Convenience wrapper around load_phits for callers that do not need rerr.
    """
    flux, _ = load_phits(path)
    return flux


def load_le_phits(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a PHITS low-energy T-Track file (log-spaced bins, 1 keV–1 MeV).

    Returns
    -------
    e_lo   : ndarray  Lower bin edges [MeV]
    e_hi   : ndarray  Upper bin edges [MeV]
    flux   : ndarray  Neutron flux [n/cm²/source]
    sigma  : ndarray  Absolute uncertainty (flux × rerr)
    """
    lines = Path(path).read_text().splitlines()
    hdr = next((ln for ln in lines if 'e-lower' in ln and 'neutron' in ln), None)
    if hdr is None:
        raise ValueError(f"No T-Track header found in {path}")
    idx = lines.index(hdr)
    e_lo, e_hi, flux_vals, rerr_vals = [], [], [], []
    for ln in lines[idx + 1 :]:
        p = ln.split()
        if len(p) < 4:
            break
        try:
            e_lo.append(float(p[0])); e_hi.append(float(p[1]))
            flux_vals.append(float(p[2])); rerr_vals.append(float(p[3]))
        except ValueError:
            break
    e_lo_arr = np.array(e_lo)
    e_hi_arr = np.array(e_hi)
    fl = np.array(flux_vals)
    return e_lo_arr, e_hi_arr, fl, fl * np.array(rerr_vals)


def load_spectrum(path: str | Path) -> np.ndarray:
    """
    Load an input neutron spectrum and normalise it so it sums to 1.

    Accepts PHITS T-Track format (header: 'e-lower ... neutron') or a plain
    two-column text file (energy, flux).  Returns shape (250,).
    """
    lines = Path(path).read_text().splitlines()
    hdr = next((ln for ln in lines if 'e-lower' in ln and 'neutron' in ln), None)
    if hdr:
        idx = lines.index(hdr)
        raw = np.array([
            float(ln.split()[2])
            for ln in lines[idx + 1 : idx + 251]
            if len(ln.split()) >= 3
        ])
    else:
        raw = np.loadtxt(path)[:250, -1]
    s = raw.sum()
    if s == 0:
        raise ValueError(f"Spectrum sums to zero: {path}")
    return raw / s
