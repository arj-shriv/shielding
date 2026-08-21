"""
Source spectrum functions for CNN training.

Each callable: f(E_MeV: ndarray[250]) → ndarray[250]
  Returns unnormalised flux values with energy-dependent log-normal noise applied.
  Callers normalise to sum=1 after random bin selection (as in generate_output_flux).

Available sources
-----------------
  tracknet10  — hardcoded piecewise6_noise fit to Ca-48 TrackNet10 (no shielding)
  concrete25  — fit to PHITS Concrete 25cm output (E_cut=10 MeV)
  steel27     — fit to PHITS Steel 27cm output    (E_cut=10 MeV)
  bpe25       — fit to PHITS BPE 25cm output      (E_cut=10 MeV)

Usage
-----
    from shielding_ml.data.source_spectra import SOURCE_FUNCTIONS
    weights = SOURCE_FUNCTIONS['concrete25'](E_MeV)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from shielding_ml.pipelines.paths import PHITS_HE
from shielding_ml.data.loaders import load_phits

_E = np.linspace(0.5, 249.5, 250)


# ── fitting primitives ────────────────────────────────────────────────────────

def _smooth_tbpl(E, A, Eb0, a1, a2, s0, Eb1, a3, s1, Eb2, a4, s2, Ec):
    x0 = np.where(E > 0, E / Eb0,  1e-30)
    x1 = np.where(E > 0, E / Eb1,  1e-30)
    x2 = np.where(E > 0, E / Eb2,  1e-30)
    return (A * x0**(-a1)
            * (1 + x0**s0)**((a1 - a2) / s0)
            * (1 + x1**s1)**((a2 - a3) / s1)
            * (1 + x2**s2)**((a3 - a4) / s2)
            * np.exp(-E / Ec))


def _piece_high(E, A_h, E_h, Eb2, alpha2, S3, Eb3, E4, S4):
    val_Eb2 = A_h * np.exp(-Eb2 / E_h)
    B2 = val_Eb2 * Eb2**alpha2
    A3 = S3 * B2
    val_Eb3 = A3 * Eb3**(-alpha2)
    A4 = S4 * val_Eb3 * np.exp(Eb3 / E4)
    return np.select(
        [E < Eb2, (E >= Eb2) & (E < Eb3), E >= Eb3],
        [A_h * np.exp(-E / E_h),
         np.where(E > 0, A3 * E**(-alpha2), 0.0),
         A4  * np.exp(-E / E4)],
    )


# ── generic fit + noisy callable builder ─────────────────────────────────────

def _make_noisy_source(fname: str, E_cut: float = 10.0):
    """
    Load a PHITS file, fit smooth_tbpl (E < E_cut) + piece_high (E >= E_cut),
    fit log-normal noise sigma(E) = C·E^a from residuals, return noisy callable.
    """
    raw, _ = load_phits(str(PHITS_HE / fname))
    F = raw / raw.sum()

    # ── low-energy fit ────────────────────────────────────────────────────────
    mk_l = (F > 0) & (_E > 0) & (_E < E_cut)
    El, Fl = _E[mk_l], F[mk_l]
    p0_l = [Fl[0], 0.8, 1.0, 0.5, 2.0, 3.0, 0.1, 2.0,
            max(E_cut * 0.7, 4.0), -0.3, 2.0, E_cut * 0.9]
    bnd_l = (
        [0,  0,   -np.inf, -np.inf, 0.1, E_cut*0.2, -np.inf, 0.1, E_cut*0.4, -np.inf, 0.1,  0],
        [np.inf, E_cut*0.2, np.inf, np.inf, 20, E_cut*0.5, np.inf, 20, E_cut, np.inf, 20, np.inf],
    )
    popt_l, _ = curve_fit(_smooth_tbpl, El, Fl, p0=p0_l, bounds=bnd_l, maxfev=2_000_000)
    A_l, Eb0_l, a1_l, a2_l, s0_l, Eb1_l, a3_l, s1_l, Eb2_l, a4_l, s2_l, Ec_l = popt_l

    # ── high-energy fit ───────────────────────────────────────────────────────
    mk_h = (F > 0) & (_E >= E_cut)
    Eh, Fh = _E[mk_h], F[mk_h]
    p0_h = [Fh[0], 15.0, min(50.0, Eh.max() * 0.4), 2.0, 1.0,
            min(130.0, Eh.max() * 0.8), 40.0, 1.0]
    bnd_h = (
        [0, 0, E_cut, 0, 0, E_cut, 0, 0],
        [np.inf, np.inf, Eh.max(), np.inf, np.inf, Eh.max(), np.inf, np.inf],
    )
    popt_h, _ = curve_fit(_piece_high, Eh, Fh, p0=p0_h, bounds=bnd_h, maxfev=500_000)
    A_h, E_h, Eb2_h, al2, S3, Eb3, E4, S4 = popt_h

    vEb2   = A_h * np.exp(-Eb2_h / E_h)
    B2     = vEb2 * Eb2_h**al2;   A3  = S3 * B2
    vEb3   = A3   * Eb3**(-al2);  A4h = S4 * vEb3 * np.exp(Eb3 / E4)

    def _base(E_arr: np.ndarray) -> np.ndarray:
        E_arr = np.asarray(E_arr, dtype=float)
        x0 = np.where(E_arr > 0, E_arr / Eb0_l, 1e-30)
        x1 = np.where(E_arr > 0, E_arr / Eb1_l, 1e-30)
        x2 = np.where(E_arr > 0, E_arr / Eb2_l, 1e-30)
        fl = (A_l * x0**(-a1_l)
              * (1 + x0**s0_l)**((a1_l - a2_l) / s0_l)
              * (1 + x1**s1_l)**((a2_l - a3_l) / s1_l)
              * (1 + x2**s2_l)**((a3_l - a4_l) / s2_l)
              * np.exp(-E_arr / Ec_l))
        fh = np.where(E_arr < Eb2_h,  A_h * np.exp(-E_arr / E_h),
             np.where(E_arr < Eb3,    np.where(E_arr > 0, A3 * E_arr**(-al2), 0.0),
                                       A4h * np.exp(-E_arr / E4)))
        return np.where(E_arr < E_cut, fl, fh)

    # ── noise: sigma(E) = C·E^a from log-residuals ────────────────────────────
    mk_all = (F > 0) & (_E > 0)
    log_res = np.abs(np.log(F[mk_all]) - np.log(_base(_E[mk_all])))
    try:
        popt_s, _ = curve_fit(
            lambda E, C, a: C * E**a, _E[mk_all], log_res,
            p0=[float(np.mean(log_res)), 0.0], maxfev=50_000,
        )
        C_s, a_s = float(popt_s[0]), float(popt_s[1])
    except Exception:
        C_s, a_s = float(np.mean(log_res)), 0.0

    def _noisy(E_arr: np.ndarray) -> np.ndarray:
        E_arr = np.asarray(E_arr, dtype=float)
        base  = _base(E_arr)
        sigma = C_s * np.where(E_arr > 0, E_arr, 1.0) ** a_s
        noise = np.exp(sigma * np.random.randn(*E_arr.shape))
        return base * noise

    return _noisy


# ── TrackNet10 — hardcoded piecewise6_noise params ───────────────────────────

def tracknet10(E: np.ndarray) -> np.ndarray:
    """Piecewise fit to TrackNet10 with energy-dependent log-normal noise."""
    E = np.asarray(E, dtype=float)

    # Low-energy (0–50 MeV): triply-broken PL × exp
    A_l   = 3.4444e+02;  Eb0   = 5.0000e+00
    a1    = -1.5799e+00; a2    = 6.5836e+00;  s0 = 5.4673e-01
    Eb1   = 5.8521e+00;  a3    = 5.7953e+00;  s1 = 9.1866e+00
    Eb2_l = 2.0000e+01;  a4    = 1.3958e+00;  s2 = 2.2124e+00
    Ec    = 5.6173e+01

    # High-energy (>=50 MeV)
    A_h   = 9.7272e-04;  E_h   = 5.3199e+01
    Eb2_h = 1.0229e+02;  alpha2= 2.2604e+00;  S3 = 1.0039e+00
    Eb3   = 1.6000e+02;  E4    = 5.0713e+01;  S4 = 9.2450e-01

    # Noise
    C_s = 1.5207e-06;  a_s = 2.2575e+00

    x0 = np.where(E > 0, E / Eb0,   1e-30)
    x1 = np.where(E > 0, E / Eb1,   1e-30)
    x2 = np.where(E > 0, E / Eb2_l, 1e-30)
    f_low = (A_l * x0**(-a1)
             * (1 + x0**s0)**((a1 - a2) / s0)
             * (1 + x1**s1)**((a2 - a3) / s1)
             * (1 + x2**s2)**((a3 - a4) / s2)
             * np.exp(-E / Ec))

    val_Eb2_h = A_h * np.exp(-Eb2_h / E_h)
    B2  = val_Eb2_h * Eb2_h**alpha2
    A3  = S3 * B2
    val_Eb3 = A3 * Eb3**(-alpha2)
    A4  = S4 * val_Eb3 * np.exp(Eb3 / E4)
    f_high = np.where(E < Eb2_h,  A_h * np.exp(-E / E_h),
             np.where(E < Eb3,    np.where(E > 0, A3 * E**(-alpha2), 0.0),
                                   A4  * np.exp(-E / E4)))

    base  = np.where(E < 50.0, f_low, f_high)
    sigma = C_s * np.where(E > 0, E, 1.0) ** a_s
    noise = np.exp(sigma * np.random.randn(*E.shape))
    return base * noise


# ── fit shielded sources on import ────────────────────────────────────────────

print('[source_spectra] Fitting Concrete 25cm ...')
concrete25 = _make_noisy_source('Concrete_25cm_1.out', E_cut=10.0)
print('[source_spectra] Fitting Concrete 45cm ...')
concrete45 = _make_noisy_source('Concrete_45cm_1.out', E_cut=10.0)
print('[source_spectra] Fitting Steel 27cm ...')
steel27    = _make_noisy_source('Steel_27cm_1.out',    E_cut=10.0)
print('[source_spectra] Fitting BPE 25cm ...')
bpe25      = _make_noisy_source('BPE_25cm.out',        E_cut=10.0)
print('[source_spectra] All sources ready.')

SOURCE_FUNCTIONS: dict[str, object] = {
    'tracknet10': tracknet10,
    'concrete25': concrete25,
    'concrete45': concrete45,
    'steel27':    steel27,
    'bpe25':      bpe25,
}
