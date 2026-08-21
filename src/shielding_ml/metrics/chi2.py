"""
Reduced chi-squared and flux MAPE.
"""

from __future__ import annotations

import numpy as np


def reduced_chi2(
    pred: np.ndarray,
    actual: np.ndarray,
    sigma: np.ndarray,
    n_params: int = 0,
    valid_mask: np.ndarray | None = None,
) -> float:
    """
    Reduced chi-squared: sum((pred - actual)^2 / sigma^2) / dof.
    Bins where sigma <= 0 or actual <= 0 are excluded automatically.
    """
    if valid_mask is None:
        valid_mask = (actual > 0) & (sigma > 0) & np.isfinite(sigma)
    n = int(valid_mask.sum())
    dof = max(n - n_params, 1)
    chi2 = float(np.sum(((pred[valid_mask] - actual[valid_mask]) / sigma[valid_mask]) ** 2))
    return chi2 / dof


def flux_mape(
    pred: np.ndarray,
    actual: np.ndarray,
    exclude_last: bool = True,
) -> float:
    """
    Mean absolute percentage error on flux, excluding bins where actual <= 0.
    If exclude_last=True, bin 249 (index 249) is excluded (as in make_dose_pdf).
    """
    n_bins = 250
    indices = np.arange(n_bins)
    if exclude_last:
        mask = (actual > 0) & (indices < n_bins - 1)
    else:
        mask = actual > 0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask]) * 100)
