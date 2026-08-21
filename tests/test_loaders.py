"""
Regression tests for PHITS and spectrum loaders.
Run from the repo root: pytest tests/
"""

import numpy as np
import pytest
from pathlib import Path

# These tests use actual data files from the repo.
# Skip if the data directory is absent (CI environment without data).

DATA_HE = Path(__file__).parents[1] / "data" / "raw" / "phits" / "high_energy"
SPECTRUM = Path(__file__).parents[1] / "data" / "raw" / "spectra" / "TrackNet10_spectrum.dat"


@pytest.mark.skipif(not DATA_HE.exists(), reason="HE PHITS data not yet migrated")
def test_load_phits_shape():
    from shielding_ml.data.loaders import load_phits
    sample = next(DATA_HE.glob("*.out"), None)
    if sample is None:
        pytest.skip("No .out files in HE directory")
    flux, rerr = load_phits(sample)
    assert flux.shape == (250,)
    assert rerr.shape == (250,)
    assert np.all(flux >= 0)


@pytest.mark.skipif(not SPECTRUM.exists(), reason="Spectrum not yet migrated")
def test_load_spectrum_normalised():
    from shielding_ml.data.loaders import load_spectrum
    spec = load_spectrum(SPECTRUM)
    assert spec.shape == (250,)
    assert abs(spec.sum() - 1.0) < 1e-10


def test_dose_midpoint_excludes_bin0():
    from shielding_ml.metrics.dose import dose_midpoint
    import numpy as np
    flux = np.ones(250)
    r_all = np.ones(250)
    d_excl = dose_midpoint(flux, r_all, excl_bin0=True)
    d_incl = dose_midpoint(flux, r_all, excl_bin0=False)
    assert d_excl == pytest.approx(249.0)
    assert d_incl == pytest.approx(250.0)


def test_dose_midpoint_max_bin():
    from shielding_ml.metrics.dose import dose_midpoint
    flux  = np.ones(250)
    r_all = np.ones(250)
    d5 = dose_midpoint(flux, r_all, excl_bin0=True, max_bin=5)
    assert d5 == pytest.approx(4.0)  # bins 1,2,3,4


def test_flux_mape_perfect():
    from shielding_ml.metrics.chi2 import flux_mape
    flux = np.ones(250)
    assert flux_mape(flux, flux) == pytest.approx(0.0)


def test_build_input_shape():
    from shielding_ml.models.inference import build_input
    spectrum = np.ones(250) / 250
    X = build_input(spectrum, thickness=25.0, density=2.3, scale=0.001)
    assert X.shape == (1, 250, 3)
