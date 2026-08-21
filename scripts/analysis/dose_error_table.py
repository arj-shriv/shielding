"""
Clean % dose error table: old TrackNet10 vs new k100s2.
Uses the same PHITS cases as run_all_inference.py (PHITS_HE auto-discovery).
"""

from __future__ import annotations
import pickle, re, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_HE, TRACKNET10_SPECTRUM, DOSE_TABLE,
)
from shielding_ml.data.constants import FLUX_SCALE, MATERIALS
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint

# ── models ────────────────────────────────────────────────────────────────────
_OLD_PKL = (Path('/Users/arjun/Python_Codes') / '20260706_TrackNet10' /
            '1D_CNN_GaussianBroadenedImpulse_MAE_ShieldingThickness_150MeV_'
            '10000samples_TrackNet10Weights_LogValues_kernel_3_'
            'CombinedMaterials_epoch20.pkl')
_NEW_PKL = (REPO_ROOT / 'models'
            / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
            / 'v1' / 'model.pkl')

OUT_PDF = REPO_ROOT / 'inference_results' / 'dose_error_comparison' / 'dose_error_table.pdf'
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ── same PHITS discovery as run_all_inference.py ──────────────────────────────
def _discover_single_cases():
    candidates: dict[tuple[str, int], Path] = {}
    for f in sorted(Path(PHITS_HE).iterdir()):
        if f.suffix not in {'.out', '.txt', ''} or f.name.startswith('.'):
            continue
        if re.search(r'_2$', f.stem):
            continue
        m = re.match(r'^([A-Za-z]+)_(\d+)cm', f.stem)
        if not m:
            continue
        mat, thick = m.group(1), int(m.group(2))
        if mat not in MATERIALS:
            continue
        is_v1 = bool(re.search(r'_1$', f.stem))
        key = (mat, thick)
        if key not in candidates or is_v1:
            candidates[key] = f
    return sorted(
        [(mat, thick, path) for (mat, thick), path in candidates.items()],
        key=lambda x: (x[0], x[1]),
    )

SINGLE_CASES = _discover_single_cases()

# ── load ──────────────────────────────────────────────────────────────────────
with open(_OLD_PKL, 'rb') as f: old_model = pickle.load(f)
with open(_NEW_PKL, 'rb') as f: new_model = pickle.load(f)
A_SPEC = load_spectrum(str(TRACKNET10_SPECTRUM))
R_ALL  = _build_r_all(DOSE_TABLE)

def cnn_dose(model, mat, thick):
    dens = MATERIALS[mat]
    B  = np.ones(250) * thick * 0.001
    B1 = np.ones(250) * dens  * 0.001
    X  = np.array([A_SPEC, B, B1]).T.reshape(1, 250, 3)
    flux = 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE
    return dose_midpoint(flux, R_ALL, excl_bin0=True)

# ── compute ───────────────────────────────────────────────────────────────────
rows = []
for mat, thick, path in SINGLE_CASES:
    phits_flux, _ = load_phits(str(path))
    if phits_flux is None:
        print(f'  skip {path.name}')
        continue
    pd = dose_midpoint(phits_flux, R_ALL, excl_bin0=True)
    od = cnn_dose(old_model, mat, thick)
    nd = cnn_dose(new_model, mat, thick)
    rows.append((mat, thick, pd, 100*(od-pd)/pd, 100*(nd-pd)/pd))
    print(f'  {mat} {thick}cm  PHITS={pd:.3e}  '
          f'old={100*(od-pd)/pd:+.1f}%  new={100*(nd-pd)/pd:+.1f}%')

# ── colour ────────────────────────────────────────────────────────────────────
def _ec(pct):
    v = abs(pct)
    if v < 20:  return '#c8e6c9'
    if v < 30:  return '#fff9c4'
    return              '#ffcdd2'

MAT_BG = {'Concrete': '#ddeeff', 'Steel': '#fde8e8', 'BPE': '#e4f4e4'}

# ── draw ──────────────────────────────────────────────────────────────────────
col_labels = ['Material', 'Thickness (cm)', 'Old model\n% error', 'New model\n% error']

cell_text, cell_colors = [], []
for mat, thick, pd, op, np_ in rows:
    cell_text.append([mat, str(thick), f'{op:+.1f}%', f'{np_:+.1f}%'])
    cell_colors.append([MAT_BG[mat], MAT_BG[mat], _ec(op), _ec(np_)])

avg_old = np.mean([abs(op) for *_, op, _ in rows])
avg_new = np.mean([abs(np_) for *_, np_ in rows])

fig, ax = plt.subplots(figsize=(8, 0.52 * len(rows) + 3.2))
ax.axis('off')

# title — moved down
fig.suptitle(
    'Model errors from PHITS dose rate (mrem/hr)',
    fontsize=13, fontweight='bold', y=0.93,
)

# avg error — just above the table
fig.text(0.5, 0.84,
         f'Average absolute dose error:   '
         f'Old model = {avg_old:.1f}%     '
         f'New model = {avg_new:.1f}%',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4ff', edgecolor='#aaaacc'))

tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               cellColours=cell_colors, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1a3a5c')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# legend
legend = [('#c8e6c9','<20%'), ('#fff9c4','20–30%'), ('#ffcdd2','>30%')]
fig.text(0.13, 0.01, '|error|:', fontsize=8)
x0 = 0.22
for color, label in legend:
    fig.add_artist(plt.Rectangle((x0, 0.005), 0.02, 0.016, color=color,
                                  transform=fig.transFigure, clip_on=False,
                                  linewidth=0.5, edgecolor='grey'))
    fig.text(x0 + 0.025, 0.01, label, fontsize=8)
    x0 += 0.17

with PdfPages(OUT_PDF) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved → {OUT_PDF}')
