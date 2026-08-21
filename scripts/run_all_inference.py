"""
Run inference for every registered model against all PHITS cases.

Outputs:
  inference_results/<model-id>/single_layer/inference.pdf
  inference_results/<model-id>/multilayer/inference.pdf

Each PDF page (one per PHITS case) contains:
  - Flux panel  : PHITS (with σ band) vs CNN (semilogy), annotated with flux MSE + MAPE
  - Ratio panel : CNN / PHITS per energy bin
  - Dose table  : PHITS dose | CNN dose | dose % error | flux MAPE

Usage:
    cd shielding-ml
    python scripts/run_all_inference.py [--only-active]
"""

import csv, json, os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_HE, PHITS_ML, DATA_REF, DOSE_TABLE, TRACKNET10_SPECTRUM,
)

parser = argparse.ArgumentParser(description='Run inference for all registered models')
parser.add_argument('--only-active', action='store_true',
                    help='Skip archived models; run only status=active')
parser.add_argument('--multilayer-only', action='store_true',
                    help='Skip single-layer PDFs; regenerate only multilayer PDFs')
parser.add_argument('--model-id', default=None,
                    help='Run inference for only this model_id')
args = parser.parse_args()

# ── PHITS case definitions ────────────────────────────────────────────────────
def _discover_single_cases():
    """Auto-discover single-layer PHITS files from PHITS_HE.

    Rules:
    - Accept .out, .txt, and no-extension files; skip .pht and hidden files
    - Skip any file whose stem ends with _2 (secondary/backup variants)
    - For a given (material, thickness), prefer the _1 variant over the base file
    - Only include materials present in MATERIALS dict
    """
    GOOD_SUFFIXES = {'.out', '.txt', ''}
    candidates: dict[tuple[str, int], Path] = {}  # (mat, thick) -> best path

    if not PHITS_HE.exists():
        return []

    for f in sorted(PHITS_HE.iterdir()):
        if f.suffix not in GOOD_SUFFIXES or f.name.startswith('.'):
            continue
        stem = f.stem
        # skip _2 variants
        if re.search(r'_2$', stem):
            continue
        m = re.match(r'^([A-Za-z]+)_(\d+)cm', stem)
        if not m:
            continue
        mat, thick = m.group(1), int(m.group(2))
        if mat not in MATERIALS:
            continue
        is_v1 = bool(re.search(r'_1$', stem))
        key = (mat, thick)
        # prefer _1 variant; if we already have a _1 don't overwrite with base
        if key not in candidates or is_v1:
            candidates[key] = f

    return sorted(
        [(mat, thick, str(path)) for (mat, thick), path in candidates.items()],
        key=lambda x: (x[0], x[1])
    )

SINGLE_CASES = _discover_single_cases()

def _discover_multilayer_cases():
    """Auto-discover .out files in PHITS_ML and parse layer info from filename."""
    cases = []
    if not PHITS_ML.exists():
        return cases
    for f in sorted(PHITS_ML.iterdir()):
        if f.suffix not in ('.out', '') or f.name.startswith('.'):
            continue
        layers = re.findall(r'([A-Za-z]+)_(\d+)cm', f.stem)
        if len(layers) >= 2 and all(m in MATERIALS for m, _ in layers):
            cases.append((str(f), [(m, int(t)) for m, t in layers]))
    return cases

MULTILAYER_CASES = _discover_multilayer_cases()

# ── Shared setup ──────────────────────────────────────────────────────────────
_R_ALL  = _build_r_all(DOSE_TABLE)
A_SPEC  = load_spectrum(TRACKNET10_SPECTRUM)
A_SPEC  = A_SPEC / A_SPEC.sum()   # ensure sum=1 (matches training convention)

VOID_FLUX, _ = load_phits(str(DATA_REF / 'No_Shielding.out'))
if VOID_FLUX is None:
    raise SystemExit('Could not parse No_Shielding.out')


def dose_excl_bin0(flux):
    return dose_midpoint(flux, _R_ALL, excl_bin0=True)


def flux_metrics(cnn, phits):
    """Returns (mse, mape) for bins 1–249 where phits > 0."""
    mask = (phits[1:249] > 0) & np.isfinite(cnn[1:249])
    if mask.sum() == 0:
        return float('nan'), float('nan')
    mse  = float(np.mean((cnn[1:249][mask] - phits[1:249][mask]) ** 2))
    mape = float(np.mean(np.abs(cnn[1:249][mask] - phits[1:249][mask])
                          / phits[1:249][mask]) * 100)
    return mse, mape


def _err_color(pct):
    v = abs(pct)
    if v < 20:  return (0.55, 0.86, 0.60, 0.75)
    if v < 30:  return (1.00, 0.93, 0.45, 0.75)
    return (0.95, 0.45, 0.45, 0.75)


def _cnn_predict(model, mat, thick, scale, spec=None):
    if spec is None:
        spec = A_SPEC
    density = MATERIALS[mat]
    B  = np.ones(250) * thick   * scale
    B1 = np.ones(250) * density * scale
    X  = np.array([spec, B, B1]).T.reshape(1, 250, 3)
    return 10 ** model.predict(X, verbose=0).flatten() * FLUX_SCALE


# ── Single-layer PDF ──────────────────────────────────────────────────────────
def make_single_layer_pdf(model, model_id, scale, val_mae, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        for mat, thick, phits_path in SINGLE_CASES:
            if not os.path.exists(phits_path):
                print(f'    [skip] not found: {phits_path}')
                continue

            phits_flux, phits_rerr = load_phits(phits_path)
            if phits_flux is None:
                print(f'    [skip] parse failed: {phits_path}')
                continue

            cnn_flux    = _cnn_predict(model, mat, thick, scale)
            phits_dose  = dose_excl_bin0(phits_flux)
            cnn_dose    = dose_excl_bin0(cnn_flux)
            dose_pct    = 100 * (cnn_dose - phits_dose) / phits_dose
            mse, mape   = flux_metrics(cnn_flux, phits_flux)
            phits_sig   = phits_flux * np.nan_to_num(phits_rerr)
            sum_cnn     = float(cnn_flux.sum())
            sum_phits   = float(phits_flux.sum())
            sum_spec    = float(A_SPEC.sum())
            mask_r      = phits_flux[1:] > 0
            avg_ratio   = float(np.mean(cnn_flux[1:][mask_r] / phits_flux[1:][mask_r]))

            fig = plt.figure(figsize=(12, 10))
            fig.patch.set_facecolor('#f8f9fa')
            gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1.2, 0.9],
                                   hspace=0.45, figure=fig)
            ax_f = fig.add_subplot(gs[0])
            ax_r = fig.add_subplot(gs[1])
            ax_t = fig.add_subplot(gs[2])

            E = E_ALL[1:]
            ax_f.semilogy(E, phits_flux[1:], color='black', linewidth=1.4,
                          label='PHITS', zorder=5)
            ax_f.fill_between(E,
                              np.clip(phits_flux[1:] - phits_sig[1:], 1e-30, None),
                              phits_flux[1:] + phits_sig[1:],
                              color='black', alpha=0.08)
            ax_f.semilogy(E, cnn_flux[1:], color='steelblue', linewidth=1.4,
                          linestyle='--', label='CNN')
            ax_f.set_xlim(1, 250)
            ax_f.set_ylim(bottom=1e-7)
            ax_f.set_xlabel('Energy (MeV)', fontsize=11)
            ax_f.set_ylabel('Neutron flux  [n/cm²/MeV/source]', fontsize=10)
            ax_f.legend(fontsize=9)
            ax_f.grid(True, which='both', alpha=0.2)
            ax_f.set_facecolor('#f0f2f5')

            # Annotation box
            mae_str = f'{val_mae:.4f}' if val_mae is not None else 'N/A'
            ann = (f'Flux MSE:          {mse:.3e}\n'
                   f'Flux MAPE:         {mape:.1f}%\n'
                   f'Avg CNN/PHITS:     {avg_ratio:.4f}\n'
                   f'sum(CNN):          {sum_cnn:.4e}\n'
                   f'sum(CNN)/sum(spec):{sum_cnn/sum_spec:.4e}\n'
                   f'sum(CNN)/sum(PHITS):{sum_cnn/sum_phits:.4f}\n'
                   f'Dose err:          {dose_pct:+.1f}%\n'
                   f'Val MAE:           {mae_str}')
            ax_f.text(0.98, 0.97, ann, transform=ax_f.transAxes,
                      fontsize=8.0, va='top', ha='right',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

            # Ratio panel
            mask  = (phits_flux[1:] > 0) & np.isfinite(cnn_flux[1:])
            ratio = np.where(mask, cnn_flux[1:] / phits_flux[1:], np.nan)
            ax_r.axhline(1, color='black', linewidth=1.0, linestyle='--', zorder=5)
            ax_r.axhspan(0.8, 1.2, alpha=0.07, color='green')
            ax_r.plot(E, ratio, color='steelblue', linewidth=1.0)
            valid = ratio[np.isfinite(ratio)]
            if len(valid):
                rlo = max(0, np.nanpercentile(valid, 2)  * 0.9)
                rhi = min(10, np.nanpercentile(valid, 98) * 1.1)
                ax_r.set_ylim(rlo, rhi)
            ax_r.set_xlim(1, 250)
            ax_r.set_xlabel('Energy (MeV)', fontsize=11)
            ax_r.set_ylabel('CNN / PHITS', fontsize=10)
            ax_r.grid(True, alpha=0.25)
            ax_r.set_facecolor('#f0f2f5')

            # Dose table
            ax_t.axis('off')
            col_labels = ['', 'Dose (1–250 MeV)  [pSv/src]', 'Dose % error', 'Flux MAPE']
            rows = [
                ['PHITS', f'{phits_dose:.4e}', '—',              '—'],
                ['CNN',   f'{cnn_dose:.4e}',   f'{dose_pct:+.1f}%', f'{mape:.1f}%'],
            ]
            cell_colors = [
                ['#2c3e50', '#d0e8ff', '#d0e8ff', '#d0e8ff'],
                ['#f0f0f0', 'white',   _err_color(dose_pct), _err_color(mape)],
            ]
            tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                             cellColours=cell_colors, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.8)
            for j in range(4):
                tbl[0, j].set_facecolor('#1a252f')
                tbl[0, j].set_text_props(color='white', fontweight='bold')
            tbl[1, 0].set_text_props(color='white', fontweight='bold')

            fig.suptitle(f'{mat}  {thick} cm  —  {model_id}',
                         fontsize=13, fontweight='bold', color='#1a252f', y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f'    {mat} {thick}cm  →  dose_err={dose_pct:+.1f}%  mape={mape:.1f}%')

    print(f'  Saved → {out_path}')


# ── Multilayer chained inference ──────────────────────────────────────────────
def _multilayer_cnn(model, layers, scale):
    """
    Mirrors Shielding_Optimization_Plot_Generator.py:
      a = layer1[20:150].sum() / void[20:150].sum()   (attenuation vs no-shield)
      b = mean( (layer2_calibrated)[20:150] / layer1_raw[20:150] )  for 3+ layers
      final = pred_last * a [* b * ...]
    b is only computed for INTERMEDIATE layers and applied to the NEXT layer.
    """
    spec          = A_SPEC.copy()
    cumulative_scale = 1.0
    prev_pred_raw = None
    inter_specs   = []

    for i, (mat, thick) in enumerate(layers):
        pred = _cnn_predict(model, mat, thick, scale, spec=spec)

        if i == 0:
            ref = VOID_FLUX[20:150].sum()
            cumulative_scale = pred[20:150].sum() / ref if ref > 0 else 1.0
        elif i < len(layers) - 1:
            denom  = np.where(prev_pred_raw[20:150] > 0, prev_pred_raw[20:150], np.nan)
            b_vals = (pred * cumulative_scale)[20:150] / denom
            cumulative_scale *= float(np.nanmean(b_vals))

        if i == 0:
            layer0_pred = pred.copy()
        prev_pred_raw = pred.copy()
        if i < len(layers) - 1:
            s    = pred.sum()
            spec = pred / s if s > 0 else pred
            inter_specs.append((spec.copy(), pred.copy(), f'{mat} {thick}cm → next', float(s)))

    return pred * cumulative_scale, layer0_pred, inter_specs


def _multilayer_cnn_no_norm(model, layers, scale):
    """Same chaining but feeds raw pred (not normalised) into each subsequent layer."""
    spec             = A_SPEC.copy()
    cumulative_scale = 1.0
    prev_pred_raw    = None

    for i, (mat, thick) in enumerate(layers):
        pred = _cnn_predict(model, mat, thick, scale, spec=spec)

        if i == 0:
            ref = VOID_FLUX[20:150].sum()
            cumulative_scale = pred[20:150].sum() / ref if ref > 0 else 1.0
        elif i < len(layers) - 1:
            denom  = np.where(prev_pred_raw[20:150] > 0, prev_pred_raw[20:150], np.nan)
            b_vals = (pred * cumulative_scale)[20:150] / denom
            cumulative_scale *= float(np.nanmean(b_vals))

        prev_pred_raw = pred.copy()
        if i < len(layers) - 1:
            spec = pred.copy()   # no normalisation — raw flux fed to next layer

    return pred * cumulative_scale


def make_multilayer_pdf(model, model_id, scale, val_mae, out_path):
    if not MULTILAYER_CASES:
        print('  No multilayer cases found — skipping')
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        for phits_path, layers in MULTILAYER_CASES:
            if not os.path.exists(phits_path):
                print(f'    [skip] not found: {phits_path}')
                continue

            phits_flux, phits_rerr = load_phits(phits_path)
            if phits_flux is None:
                print(f'    [skip] parse failed: {phits_path}')
                continue

            layer_str      = ' → '.join(f'{m} {t} cm' for m, t in layers)
            cnn_flux, layer0_flux, inter_specs = _multilayer_cnn(model, layers, scale)
            cnn_no_norm    = _multilayer_cnn_no_norm(model, layers, scale)
            mask_r      = phits_flux[1:] > 0
            avg_ratio   = float(np.mean(cnn_flux[1:][mask_r] / phits_flux[1:][mask_r]))
            cnn_corr    = cnn_flux / avg_ratio
            phits_dose  = dose_excl_bin0(phits_flux)
            cnn_dose    = dose_excl_bin0(cnn_flux)
            corr_dose   = dose_excl_bin0(cnn_corr)
            dose_pct    = 100 * (cnn_dose  - phits_dose) / phits_dose
            corr_pct    = 100 * (corr_dose - phits_dose) / phits_dose
            mse, mape   = flux_metrics(cnn_flux, phits_flux)
            phits_sig   = phits_flux * np.nan_to_num(phits_rerr)
            sum_cnn     = float(cnn_flux.sum())
            sum_phits   = float(phits_flux.sum())
            sum_spec    = float(A_SPEC.sum())

            fig = plt.figure(figsize=(12, 13))
            fig.patch.set_facecolor('#f8f9fa')
            gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1.2, 1.2, 0.9],
                                   hspace=0.5, figure=fig)
            ax_f = fig.add_subplot(gs[0])
            ax_r = fig.add_subplot(gs[1])
            ax_s = fig.add_subplot(gs[2])
            ax_t = fig.add_subplot(gs[3])

            E = E_ALL[1:]
            ax_f.semilogy(E, VOID_FLUX[1:], color='#1565C0', linewidth=1.2,
                          linestyle=':', alpha=0.7, label='Unshielded')
            ax_f.semilogy(E, phits_flux[1:], color='black', linewidth=1.4,
                          label='PHITS', zorder=5)
            ax_f.fill_between(E,
                              np.clip(phits_flux[1:] - phits_sig[1:], 1e-30, None),
                              phits_flux[1:] + phits_sig[1:],
                              color='black', alpha=0.08)
            ax_f.semilogy(E, cnn_flux[1:], color='darkorange', linewidth=1.4,
                          linestyle='--', label=f'CNN chained  (dose {dose_pct:+.1f}%)')
            ax_f.semilogy(E, cnn_corr[1:], color='steelblue', linewidth=1.4,
                          linestyle=':', label=f'CNN ÷ avg_ratio={avg_ratio:.3f}  (dose {corr_pct:+.1f}%)')
            for k, (sp, pred_raw, lbl, raw_sum) in enumerate(inter_specs):
                ax_f.semilogy(E_ALL[1:], pred_raw[1:], color='#7B1FA2', linewidth=1.2,
                              linestyle='-.', alpha=0.7,
                              label=f'pred after {lbl.split(" →")[0]} (raw, sum={raw_sum:.3e})')
            no_norm_dose = dose_excl_bin0(cnn_no_norm)
            no_norm_pct  = 100 * (no_norm_dose - phits_dose) / phits_dose
            ax_f.semilogy(E_ALL[1:], cnn_no_norm[1:], color='#2E7D32', linewidth=1.2,
                          linestyle='--', alpha=0.8,
                          label=f'CNN no-norm  (dose {no_norm_pct:+.1f}%)')
            ax_f.set_xlim(1, 250)
            ax_f.set_ylim(bottom=1e-8)
            ax_f.set_xlabel('Energy (MeV)', fontsize=11)
            ax_f.set_ylabel('Neutron flux  [n/cm²/MeV/source]', fontsize=10)
            ax_f.legend(fontsize=9)
            ax_f.grid(True, which='both', alpha=0.2)
            ax_f.set_facecolor('#f0f2f5')

            mae_str = f'{val_mae:.4f}' if val_mae is not None else 'N/A'
            ann = (f'Flux MSE:           {mse:.3e}\n'
                   f'Flux MAPE:          {mape:.1f}%\n'
                   f'Avg CNN/PHITS:      {avg_ratio:.4f}\n'
                   f'sum(CNN):           {sum_cnn:.4e}\n'
                   f'sum(CNN)/sum(spec): {sum_cnn/sum_spec:.4e}\n'
                   f'sum(CNN)/sum(PHITS):{sum_cnn/sum_phits:.4f}\n'
                   f'Dose err (raw):     {dose_pct:+.1f}%\n'
                   f'Dose err (corr):    {corr_pct:+.1f}%\n'
                   f'Val MAE:            {mae_str}')
            ax_f.text(0.98, 0.97, ann, transform=ax_f.transAxes,
                      fontsize=8.0, va='top', ha='right',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

            mask  = (phits_flux[1:] > 0) & np.isfinite(cnn_flux[1:])
            ratio      = np.where(mask, cnn_flux[1:]  / phits_flux[1:], np.nan)
            ratio_corr = np.where(mask, cnn_corr[1:]  / phits_flux[1:], np.nan)
            ax_r.axhline(1, color='black', linewidth=1.0, linestyle='--', zorder=5)
            ax_r.axhspan(0.8, 1.2, alpha=0.07, color='green')
            ax_r.plot(E, ratio,      color='darkorange', linewidth=1.0, label='CNN chained / PHITS')
            ax_r.plot(E, ratio_corr, color='steelblue',  linewidth=1.0, linestyle=':', label='Corrected / PHITS')
            for k, (sp, pred_raw, lbl, raw_sum) in enumerate(inter_specs):
                ratio_raw = np.where(mask, pred_raw[1:] / phits_flux[1:], np.nan)
                ax_r.plot(E, ratio_raw, color='#7B1FA2', linewidth=1.0, linestyle='-.',
                          label=f'pred {lbl.split(" →")[0]} raw / PHITS')
            valid = ratio[np.isfinite(ratio)]
            if len(valid):
                rlo = max(0, np.nanpercentile(valid, 2)  * 0.9)
                rhi = min(10, np.nanpercentile(valid, 98) * 1.1)
                ax_r.set_ylim(rlo, rhi)
            ax_r.set_xlim(1, 250)
            ax_r.set_xlabel('Energy (MeV)', fontsize=11)
            ax_r.set_ylabel('CNN / PHITS', fontsize=10)
            ax_r.legend(fontsize=8)
            ax_r.grid(True, alpha=0.25)
            ax_r.set_facecolor('#f0f2f5')

            # ── Intermediate normalised spectra panel ─────────────────────────
            ax_s.semilogy(E_ALL, A_SPEC, color='#555555', lw=1.4, linestyle=':',
                          label=f'TrackNet10 (source, sum={A_SPEC.sum():.4f})')
            sp_colors = ['#E53935', '#1565C0', '#2E7D32', '#6A1B9A']
            for k, (sp, pred_raw, lbl, raw_sum) in enumerate(inter_specs):
                ax_s.semilogy(E_ALL, sp, color=sp_colors[k % len(sp_colors)],
                              lw=1.4, label=f'spec after {lbl}  (pred.sum()={raw_sum:.4e})')
            ax_s.set_xlim(0, 250)
            ax_s.set_ylim(bottom=1e-7)
            ax_s.set_xlabel('Energy (MeV)', fontsize=10)
            ax_s.set_ylabel('Normalised flux  (sum=1)', fontsize=10)
            ax_s.set_title('Input spectra passed between layers', fontsize=10, pad=4)
            ax_s.legend(fontsize=8)
            ax_s.grid(True, which='both', alpha=0.2)
            ax_s.set_facecolor('#f0f2f5')

            ax_t.axis('off')
            col_labels = ['', 'Dose (1–250 MeV)  [pSv/src]', 'Dose % error', 'Flux MAPE']
            rows = [
                ['PHITS',      f'{phits_dose:.4e}', '—',               '—'],
                ['CNN raw',    f'{cnn_dose:.4e}',   f'{dose_pct:+.1f}%',  f'{mape:.1f}%'],
                ['CNN corr',   f'{corr_dose:.4e}',  f'{corr_pct:+.1f}%',  '—'],
            ]
            cell_colors = [
                ['#2c3e50', '#d0e8ff', '#d0e8ff', '#d0e8ff'],
                ['#f0f0f0', 'white',   _err_color(dose_pct), _err_color(mape)],
                ['#e8f4e8', 'white',   _err_color(corr_pct), '#f0f0f0'],
            ]
            tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                             cellColours=cell_colors, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            for j in range(4):
                tbl[0, j].set_facecolor('#1a252f')
                tbl[0, j].set_text_props(color='white', fontweight='bold')
            tbl[1, 0].set_text_props(fontweight='bold')
            tbl[2, 0].set_text_props(fontweight='bold')

            layer0_sum = float(layer0_flux.sum())
            a_factor   = layer0_flux[20:150].sum() / VOID_FLUX[20:150].sum()
            info = (f'a = layer0[20:150].sum()/void[20:150].sum() = {a_factor:.4f}   |   '
                    f'avg CNN/PHITS: {avg_ratio:.4f}   |   '
                    f'sum(CNN layer 0) / sum(PHITS): {layer0_sum/sum_phits:.4f}   |   '
                    f'sum(CNN): {sum_cnn:.4e}   |   sum(PHITS): {sum_phits:.4e}')
            ax_t.text(0.5, -0.08, info, transform=ax_t.transAxes,
                      fontsize=7.5, ha='center', va='top', color='#333333',
                      fontfamily='monospace')

            fig.suptitle(f'Multilayer: {layer_str}  —  {model_id}',
                         fontsize=12, fontweight='bold', color='#1a252f', y=1.01)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f'    {layer_str}  →  dose_err={dose_pct:+.1f}%  mape={mape:.1f}%')
            for k, (sp, pred_raw, lbl, raw_sum) in enumerate(inter_specs):
                print(f'      [purple] pred after {lbl.split(" →")[0]}:  sum={raw_sum:.4e}  dose={dose_excl_bin0(pred_raw):.4e}')
            print(f'      [orange] CNN chained (final):    sum={sum_cnn:.4e}  dose={cnn_dose:.4e}')

    print(f'  Saved → {out_path}')


# ── Main loop ─────────────────────────────────────────────────────────────────
registry_path = REPO_ROOT / 'models' / 'registry.csv'

with open(registry_path, newline='') as f:
    models = list(csv.DictReader(f))

RESULTS_DIR = REPO_ROOT / 'inference_results'

for row in models:
    model_id   = row['model_id']
    model_path = row['path']
    status     = row['status']
    scale_str  = row['input_scale']

    if model_path == 'null' or status == 'legacy':
        print(f'[skip] {model_id}  (legacy/no path)')
        continue
    if args.model_id and model_id != args.model_id:
        continue
    if args.only_active and status != 'active':
        print(f'[skip] {model_id}  (status={status})')
        continue
    if scale_str in ('unknown', ''):
        print(f'[skip] {model_id}  (unknown scale)')
        continue

    full_path = REPO_ROOT / model_path
    if not full_path.exists():
        print(f'[skip] {model_id}  (pkl not found: {full_path})')
        continue

    scale = float(scale_str)

    # Try to read val_mae from metadata
    val_mae = None
    meta_path = full_path.parent / 'metadata.json'
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            val_mae = meta.get('val_mae')
        except Exception:
            pass

    print(f'\n{"="*60}')
    print(f'Model: {model_id}  (scale={scale}  val_mae={val_mae})')
    print(f'{"="*60}')

    model = load_model(str(full_path))

    sl_pdf = str(RESULTS_DIR / model_id / 'single_layer' / 'inference.pdf')
    ml_pdf = str(RESULTS_DIR / model_id / 'multilayer'   / 'inference.pdf')

    if not args.multilayer_only:
        print('  → Single-layer')
        make_single_layer_pdf(model, model_id, scale, val_mae, sl_pdf)

    print('  → Multilayer')
    make_multilayer_pdf(model, model_id, scale, val_mae, ml_pdf)

print('\nAll done.')
