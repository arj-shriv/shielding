"""
Generate images_for_paper/tracknet10_spectrum.png

Fits the normalised TrackNet10 spectrum (A_SPEC from TrackNet10_spectrum.dat)
with the same two-stage piecewise approach used in analysis/parse_usrtrack.ipynb:
  - Low energy  (E < 50 MeV) : triply-broken power law × exp
  - High energy (E ≥ 50 MeV) : exp → power law → exp
  - Noise model               : σ(E) = C·E^a from log-residuals
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_spectrum
from shielding_ml.data.constants import E_ALL
from shielding_ml.pipelines.paths import TRACKNET10_SPECTRUM

OUT  = Path(__file__).parent / 'tracknet10_spectrum.png'
E    = E_ALL.copy()
E_CUT = 50.0

# ── load normalised spectrum ───────────────────────────────────────────────────
A = load_spectrum(str(TRACKNET10_SPECTRUM))   # sum = 1

# ── fitting functions (identical to notebook cell 4bd2188c) ───────────────────
def smooth_tbpl(E, A_, Eb0, a1, a2, s0, Eb1, a3, s1, Eb2, a4, s2, Ec):
    x0 = np.where(E > 0, E / Eb0,  1e-30)
    x1 = np.where(E > 0, E / Eb1,  1e-30)
    x2 = np.where(E > 0, E / Eb2,  1e-30)
    return (A_
            * x0**(-a1)
            * (1 + x0**s0)**((a1 - a2) / s0)
            * (1 + x1**s1)**((a2 - a3) / s1)
            * (1 + x2**s2)**((a3 - a4) / s2)
            * np.exp(-E / Ec))

def piece_high(E, A_h, E_h, Eb2, alpha2, S3, Eb3, E4, S4):
    val_Eb2 = A_h * np.exp(-Eb2 / E_h)
    B2 = val_Eb2 * Eb2**alpha2;  A3 = S3 * B2
    val_Eb3 = A3 * Eb3**(-alpha2);  A4 = S4 * val_Eb3 * np.exp(Eb3 / E4)
    return np.select(
        [E < Eb2, (E >= Eb2) & (E < Eb3), E >= Eb3],
        [A_h * np.exp(-E / E_h),
         np.where(E > 0, A3 * E**(-alpha2), 0.0),
         A4 * np.exp(-E / E4)],
    )

# ── low-energy fit (E < 50 MeV) ───────────────────────────────────────────────
mk_l = (A > 0) & (E > 0) & (E < E_CUT)
El, Fl = E[mk_l], A[mk_l]
p0_l = [Fl[0], 2.0, 1.0, 0.5, 2.0, 8.0, 0.1, 2.0, 25.0, -0.3, 2.0, 40.0]
bnd_l = (
    [0,  0,  -np.inf,-np.inf,0.1,  5,-np.inf,0.1,20,-np.inf,0.1, 0],
    [np.inf,5, np.inf, np.inf,20, 20, np.inf,20, 50, np.inf,20,np.inf],
)
popt_l, _ = curve_fit(smooth_tbpl, El, Fl, p0=p0_l, bounds=bnd_l, maxfev=1_000_000)
A_l,Eb0_l,a1_l,a2_l,s0_l,Eb1_l,a3_l,s1_l,Eb2_l,a4_l,s2_l,Ec_l = popt_l
print(f'Low-E fit done: Eb0={Eb0_l:.2f}  Eb1={Eb1_l:.2f}  Eb2={Eb2_l:.2f}  Ec={Ec_l:.2f}')

# ── high-energy fit (E ≥ 50 MeV) ─────────────────────────────────────────────
mk_h = (A > 0) & (E >= E_CUT)
Eh, Fh = E[mk_h], A[mk_h]
p0_h = [Fh[0], 47.0, 100.0, 2.0, 1.0, 160.0, 50.0, 1.0]
bnd_h = ([0, 0, E_CUT, 0, 0, E_CUT, 0, 0],
         [np.inf, np.inf, E[-1], np.inf, np.inf, E[-1], np.inf, np.inf])
popt_h, _ = curve_fit(piece_high, Eh, Fh, p0=p0_h, bounds=bnd_h, maxfev=500_000)
A_h_v,E_h_v,Eb2_h,alpha2,S3,Eb3,E4,S4 = popt_h
val_Eb2_h = A_h_v * np.exp(-Eb2_h / E_h_v)
B2  = val_Eb2_h * Eb2_h**alpha2;  A3 = S3 * B2
val_Eb3 = A3 * Eb3**(-alpha2);  A4h = S4 * val_Eb3 * np.exp(Eb3 / E4)
print(f'High-E fit done: E_h={E_h_v:.2f}  Eb2={Eb2_h:.2f}  Eb3={Eb3:.2f}')

# ── combined piecewise function ───────────────────────────────────────────────
def piecewise(E_arr):
    E_arr = np.asarray(E_arr, dtype=float)
    x0 = np.where(E_arr > 0, E_arr / Eb0_l, 1e-30)
    x1 = np.where(E_arr > 0, E_arr / Eb1_l, 1e-30)
    x2 = np.where(E_arr > 0, E_arr / Eb2_l, 1e-30)
    fl = (A_l * x0**(-a1_l)
          * (1 + x0**s0_l)**((a1_l - a2_l) / s0_l)
          * (1 + x1**s1_l)**((a2_l - a3_l) / s1_l)
          * (1 + x2**s2_l)**((a3_l - a4_l) / s2_l)
          * np.exp(-E_arr / Ec_l))
    fh = np.where(E_arr < Eb2_h,  A_h_v * np.exp(-E_arr / E_h_v),
         np.where(E_arr < Eb3,    np.where(E_arr > 0, A3 * E_arr**(-alpha2), 0.0),
                                   A4h   * np.exp(-E_arr / E4)))
    return np.where(E_arr < E_CUT, fl, fh)

# ── noise model: σ(E) = C·E^a from log-residuals ────────────────────────────
mk_all  = (A > 0) & (E > 0)
log_res = np.log(A[mk_all]) - np.log(piecewise(E[mk_all]))
popt_s, _ = curve_fit(lambda Ev, C, a: C * Ev**a, E[mk_all], np.abs(log_res),
                       p0=[np.mean(np.abs(log_res)), 0.0], maxfev=50_000)
C_s, a_s = popt_s
print(f'Noise model: σ(E) = {C_s:.4e} · E^{a_s:.4f}')

# ── evaluate fit and noise band on smooth grid ────────────────────────────────
# Normalise piecewise to sum=1 over the same 250 discrete bins as A
_norm  = piecewise(E).sum()
E_sm   = np.linspace(0.5, 249.5, 1000)
f_sm   = piecewise(E_sm) / _norm
sig_sm = C_s * np.where(E_sm > 0, E_sm, 1.0) ** a_s

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.stairs(A, np.append(E - 0.5, 249.5),
          color='#2c6fad', lw=1.4, alpha=0.80, label='TrackNet10 spectrum')
ax.plot(E_sm, f_sm,
        color='#c0392b', lw=2.0, ls='--', zorder=4,
        label='Piecewise fit  (triply-broken PL × exp  |  exp/PL/exp)')
ax.fill_between(E_sm, f_sm * np.exp(-sig_sm), f_sm * np.exp(sig_sm),
                color='#c0392b', alpha=0.15, zorder=2,
                label=r'$\pm\sigma(E)$ noise envelope')
ax.axvline(E_CUT, color='#666', lw=0.9, ls=':', alpha=0.7,
           label=f'Fit boundary ({E_CUT:.0f} MeV)')

ax.set_yscale('log')
ax.set_xlabel('Neutron energy (MeV)', fontsize=12)
ax.set_ylabel('Normalised fluence (a.u.)', fontsize=12)
ax.set_title('TrackNet10 source spectrum — piecewise fit', fontsize=13, fontweight='bold')
ax.set_xlim(0, 250)
ax.legend(fontsize=9.5)
ax.grid(True, which='both', alpha=0.22)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {OUT}')
