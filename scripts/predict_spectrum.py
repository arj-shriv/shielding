"""
Predict the transmitted neutron spectrum for a single shielding case and plot it.

Usage:
    python scripts/predict_spectrum.py \\
        --material Concrete --thickness 75 \\
        --model-dir models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/v1 \\
        [--phits path/to/file.out]   # optional PHITS overlay
        [--out output.png]           # default: <material>_<thick>cm_<model-id>.png
        [--scale 0.001]              # input_scale (default 0.001)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── repo paths ───────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import TRACKNET10_SPECTRUM, DOSE_TABLE, PHITS_HE, DATA_REF

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--material',   required=True, choices=list(MATERIALS.keys()),
               help='Material name: Concrete, Steel, BPE')
p.add_argument('--thickness',  required=True, type=float, help='Thickness in cm')
p.add_argument('--model-dir',  required=True,
               help='Path to model directory (must contain model.pkl)')
p.add_argument('--phits',      default=None,
               help='Optional path to PHITS .out file for overlay')
p.add_argument('--out',        default=None,
               help='Output PNG path (default: <material>_<thick>cm_<model>.png)')
p.add_argument('--scale',      type=float, default=0.001, help='Input scale (default 0.001)')
args = p.parse_args()

# ── resolve model path ────────────────────────────────────────────────────────
model_dir = Path(args.model_dir)
if not model_dir.is_absolute():
    model_dir = REPO / model_dir
pkl = model_dir / 'model.pkl'
if not pkl.exists():
    # try one level up
    pkl = model_dir / 'v1' / 'model.pkl'
if not pkl.exists():
    sys.exit(f'ERROR: model.pkl not found in {model_dir}')

# ── load model + constants ────────────────────────────────────────────────────
model     = load_model(str(pkl))
A_SPEC    = load_spectrum(TRACKNET10_SPECTRUM)
R_ALL     = _build_r_all(DOSE_TABLE)
E         = np.linspace(0.5, 249.5, 250)
void_flux, _ = load_phits(str(DATA_REF / 'No_Shielding.out'))
void_dose    = dose_midpoint(void_flux, R_ALL, excl_bin0=True) if void_flux is not None else None

# ── CNN prediction ────────────────────────────────────────────────────────────
density = MATERIALS[args.material]
B  = np.ones(250) * args.thickness * args.scale
B1 = np.ones(250) * density        * args.scale
X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
log_pred   = model.predict(X, verbose=0).flatten()
cnn_flux   = 10**log_pred * FLUX_SCALE
cnn_dose   = dose_midpoint(cnn_flux, R_ALL, excl_bin0=True)
a_factor   = cnn_flux[20:150].sum() / void_flux[20:150].sum() if void_flux is not None else None

# ── optional PHITS overlay ────────────────────────────────────────────────────
phits_flux = None
phits_dose = None
if args.phits:
    phits_path = Path(args.phits)
else:
    # auto-find: prefer _1 variant (matches what run_all_inference uses), then base
    stem = f'{args.material}_{int(args.thickness)}cm'
    candidates = [
        PHITS_HE / f'{stem}_1.out',
        PHITS_HE / f'{stem}.out',
    ]
    phits_path = next((c for c in candidates if c.exists()), None)

if phits_path and phits_path.exists():
    phits_flux, _ = load_phits(str(phits_path))
    if phits_flux is not None:
        phits_dose = dose_midpoint(phits_flux, R_ALL, excl_bin0=True)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                         gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
ax_f, ax_r = axes

title = f'{args.material} {args.thickness:.0f} cm  —  {model_dir.parent.name}'
fig.suptitle(title, fontsize=12, fontweight='bold')

# flux panel
if void_flux is not None:
    ax_f.semilogy(E, void_flux, color='#888888', lw=1.2, linestyle=':',
                  label=f'Unshielded  (dose={void_dose:.3e} mrem/h)')
ax_f.semilogy(E, cnn_flux, color='#E53935', lw=1.8, label=f'CNN  (dose={cnn_dose:.3e} mrem/h)')
if phits_flux is not None:
    ax_f.semilogy(E, phits_flux, color='#1565C0', lw=1.8, linestyle='--',
                  label=f'PHITS  (dose={phits_dose:.3e} mrem/h)')
    dose_err = 100 * (cnn_dose - phits_dose) / phits_dose
    ax_f.text(0.98, 0.97, f'Dose err: {dose_err:+.1f}%',
              transform=ax_f.transAxes, ha='right', va='top',
              fontsize=10, color=('#C62828' if abs(dose_err) > 20 else '#2E7D32'),
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax_f.set_ylabel('Flux (n/cm²/source/MeV)', fontsize=11)
ax_f.legend(fontsize=10)
ax_f.grid(True, which='both', alpha=0.2)
ax_f.set_xlim(0, 250)
ax_f.set_xticklabels([])

# ratio panel (CNN / PHITS)
if phits_flux is not None:
    mask = phits_flux[1:] > 0
    ratio = np.where(mask, cnn_flux[1:] / phits_flux[1:], np.nan)
    ax_r.plot(E[1:], ratio, color='#7B1FA2', lw=1.2)
    ax_r.axhline(1, color='k', lw=0.8, linestyle='--')
    ax_r.set_ylabel('CNN / PHITS', fontsize=10)
    ax_r.set_ylim(0, max(3, np.nanpercentile(ratio, 98) * 1.2))
    ax_r.grid(True, alpha=0.2)
else:
    ax_r.text(0.5, 0.5, 'No PHITS reference available',
              transform=ax_r.transAxes, ha='center', va='center',
              fontsize=10, color='gray')
    ax_r.set_ylabel('CNN / PHITS', fontsize=10)
    ax_r.grid(True, alpha=0.2)

ax_r.set_xlabel('Energy (MeV)', fontsize=11)
ax_r.set_xlim(0, 250)

# dose + a-factor annotation box (bottom of flux panel)
dose_txt = f'CNN dose: {cnn_dose:.3e} mrem/h'
if phits_dose:
    dose_txt += f'     PHITS dose: {phits_dose:.3e} mrem/h     err: {dose_err:+.1f}%'
if a_factor is not None:
    dose_txt += f'     a = pred[20:150].sum()/void[20:150].sum() = {a_factor:.4f}'
ax_f.annotate(dose_txt, xy=(0.01, 0.02), xycoords='axes fraction',
              fontsize=9, color='#333333', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', alpha=0.9))

# ── save ──────────────────────────────────────────────────────────────────────
if args.out:
    out_path = Path(args.out)
else:
    model_id = model_dir.parent.name
    out_path = Path(f'{args.material}_{int(args.thickness)}cm_{model_id}.png')

plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path.resolve()}')
if phits_dose:
    print(f'  PHITS dose : {phits_dose:.4e} mrem/h')
print(f'  CNN   dose : {cnn_dose:.4e} mrem/h')
if phits_dose:
    print(f'  Dose error : {dose_err:+.2f}%')
