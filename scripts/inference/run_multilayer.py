"""
Multi-layer (compound shield) inference.

Strategy: chain the CNN — normalize each layer's output and feed it as the
input spectrum into the next layer, using the same model throughout.

  Layer 1:  input_spec → CNN(mat1, thick1) → pred1
  Layer 2:  normalise(pred1) → CNN(mat2, thick2) → pred_final

The compound PHITS file is auto-parsed from its basename, e.g.
  Concrete_45cm_Steel_33cm.out  →  [(Concrete, 45), (Steel, 33)]

Bi-modeling mode (--bimodel):
  Layer 1 uses the standard model; layer 2 uses a source-matched model
  (default: v_concrete25 — trained with a Concrete 25cm output as source).
  This corrects the amplitude calibration when the layer-2 input spectrum
  looks like a shielded-concrete spectrum rather than TrackNet10.

Usage (default — runs both built-in compound cases):
    python scripts/inference/run_multilayer.py

Bi-modeling:
    python scripts/inference/run_multilayer.py --bimodel
    python scripts/inference/run_multilayer.py --bimodel steel27

Specify custom model(s), PHITS file(s), and/or input spectrum:
    python scripts/inference/run_multilayer.py \\
        --models 20260714/k25_k11_k3_d256_d64_mae_e25_bins50_240:0.001 \\
                 20260713/k25_k11_k3_d256_d64_mae_e25_scale0p1:0.1 \\
        --phits  data/raw/phits/multilayer/Concrete_45cm_Steel_33cm.out \\
                 data/raw/phits/multilayer/Concrete_73cm_BPE_27cm.out \\
        --spectrum data/raw/spectra/TrackNet10_spectrum.dat \\
        --out runs/multilayer/multilayer_comparison.pdf

--models  one or more  <folder>:<scale>  pairs (scale defaults to 0.001 if omitted)
          folder is relative to the repo root
--phits   one or more compound PHITS files (relative to repo root or absolute)
--spectrum  path to the input spectrum file
--out     output PDF path
--bimodel [SOURCE]  enable bi-modeling: layer 2 uses the source-matched model
          SOURCE is one of: concrete25, steel27, bpe25  (default: concrete25)
"""

import os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from shielding_ml.data.loaders import load_phits, load_spectrum
from shielding_ml.data.constants import MATERIALS, FLUX_SCALE, E_ALL, MODEL_PKL_NAME
from shielding_ml.metrics.dose import _build_r_all, dose_midpoint
from shielding_ml.models.inference import load_model
from shielding_ml.pipelines.paths import (
    REPO_ROOT, PHITS_ML, PHITS_HE, DATA_REF, TRACKNET10_SPECTRUM, DOSE_TABLE, RUNS_DIR,
)

MODEL_COLORS = ['steelblue', 'darkorange', 'seagreen', 'orchid', 'crimson',
                'saddlebrown', 'deeppink', 'teal']

_K100S2_DIR     = REPO_ROOT / 'models' / 'k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240'
_BIMODEL_SOURCES = {'concrete25', 'steel27', 'bpe25'}

DEFAULT_L1_FOLDER = str(_K100S2_DIR / 'v1')
DEFAULT_L1_SCALE  = 0.001

DEFAULT_PHITS = sorted(
    str(p) for p in PHITS_ML.glob('*.out')
)

DEFAULT_SPECTRUM = str(TRACKNET10_SPECTRUM)
DEFAULT_VOID     = str(DATA_REF / 'No_Shielding.out')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Multi-layer chained CNN inference for compound shielding')
parser.add_argument(
    '--layer1', metavar='FOLDER[:SCALE]', default=None,
    help=f'Layer-1 model folder (relative to repo root) with optional :scale. '
         f'Default: k100s2 v1 at scale {DEFAULT_L1_SCALE}.')
parser.add_argument(
    '--models', nargs='+', metavar='FOLDER[:SCALE]',
    help='(Advanced) Compare multiple layer-1 models in one PDF. '
         'Overrides --layer1 when both are given.')
parser.add_argument(
    '--phits', nargs='+', metavar='PATH',
    help='Compound PHITS file(s) to run inference on')
parser.add_argument(
    '--spectrum', metavar='PATH', default=None,
    help='Input spectrum file (T-Track or plain txt, relative to repo root or absolute)')
parser.add_argument(
    '--void', metavar='PATH', default=None,
    help='No-shielding PHITS T-Track file used as unshielded reference')
parser.add_argument(
    '--out', metavar='PATH', default=None,
    help='Output PDF path')
parser.add_argument(
    '--output-dir', metavar='DIR', default=None,
    help='Output directory. Defaults to inference_results/bi_model_multilayer '
         'when --bimodel is set, otherwise runs/multilayer/.')
parser.add_argument(
    '--bimodel', nargs='?', const='concrete25', default=None,
    metavar='SOURCE',
    help='Enable bi-modeling: layer 1 uses --layer1 model (TrackNet10-trained), '
         'layer 2 uses a source-matched k100s2 model. '
         'SOURCE ∈ {concrete25, steel27, bpe25} (default: concrete25).')
parser.add_argument(
    '--transfer', nargs='?', const='transfer_concrete45_mlp', default=None,
    metavar='FOLDER',
    help='Use a transfer-learned head for layer 2. FOLDER is the subfolder name '
         'inside the k100s2 models dir (default: transfer_concrete45_mlp). '
         'Output goes to inference_results/<FOLDER>/.')
args = parser.parse_args()

def _abspath(p):
    return p if os.path.isabs(p) else str(REPO_ROOT / p)

# Resolve output dir and PDF path
if args.output_dir:
    out_dir = os.path.abspath(args.output_dir)
elif args.transfer:
    out_dir = str(REPO_ROOT / 'inference_results' / args.transfer)
elif args.bimodel is not None:
    out_dir = str(REPO_ROOT / 'inference_results' / 'bi_model_multilayer')
else:
    out_dir = str(RUNS_DIR / 'multilayer')
os.makedirs(out_dir, exist_ok=True)

if args.out:
    out_pdf = _abspath(args.out)
else:
    out_pdf = os.path.join(out_dir, 'multilayer_comparison.pdf')
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

# Resolve model list
def _parse_model_token(token):
    if ':' in token:
        folder, scale_str = token.rsplit(':', 1)
        scale = float(scale_str)
    else:
        folder, scale = token, DEFAULT_L1_SCALE
    label = os.path.basename(folder.rstrip('/'))
    return label, _abspath(folder), scale

if args.models:
    model_defs = [_parse_model_token(t) for t in args.models]
elif args.layer1:
    model_defs = [_parse_model_token(args.layer1)]
else:
    # default: k100s2 v1
    model_defs = [('k100s2-v1', DEFAULT_L1_FOLDER, DEFAULT_L1_SCALE)]

phits_files  = [_abspath(p) for p in (args.phits    or DEFAULT_PHITS)]
spectrum_src = _abspath(args.spectrum) if args.spectrum else DEFAULT_SPECTRUM
void_src     = _abspath(args.void)     if args.void     else DEFAULT_VOID

# ── ICRP dose conversion ─────────────────────────────────────────────────────
_R_ALL = _build_r_all(DOSE_TABLE)

def dose_excl_bin0(flux):
    return dose_midpoint(flux, _R_ALL, excl_bin0=True)

def _err_color(pct):
    v = abs(pct)
    if v < 20:  return (0.55, 0.86, 0.60, 0.7)
    if v < 30:  return (1.00, 0.93, 0.45, 0.7)
    return (0.95, 0.45, 0.45, 0.7)

def parse_layers(filepath):
    """'Concrete_45cm_Steel_33cm.out' → [('Concrete', 45), ('Steel', 33)]"""
    name = os.path.basename(filepath).replace('.out', '')
    return [(m, int(t)) for m, t in re.findall(r'([A-Za-z]+)_(\d+)cm', name)]

# ── Chained inference ────────────────────────────────────────────────────────
def multilayer_cnn(model, layers, scale, input_spec, void_flux, model_layer2=None):
    """
    Chain CNN through each layer with attenuation-factor scaling (v1/v2 approach).

    Layer 1:
        pred1 = CNN(input_spec, mat1, thick1)
        a = sum(pred1[20:150]) / sum(void_flux[20:150])   ← dimensionless, <1
        spec2 = pred1 / pred1.sum()                        ← shape for layer 2

    Layer 2 (final for 2-layer case):
        pred2 = CNN(spec2, mat2, thick2)          ← model_layer2 if bi-modeling
        final = pred2 * a

    model_layer2: if not None, used instead of model for all layers after the first
                  (bi-modeling mode).
    """
    spec             = input_spec.copy()
    cumulative_scale = 1.0
    prev_pred_raw    = None

    for i, (mat, thick) in enumerate(layers):
        active_model = model if (i == 0 or model_layer2 is None) else model_layer2
        density = MATERIALS[mat]
        B  = np.ones(250) * thick   * scale
        B1 = np.ones(250) * density * scale
        X  = np.array([spec, B, B1]).T.reshape(1, 250, 3)
        log_pred = active_model.predict(X, verbose=0).flatten()
        pred = 10**log_pred * FLUX_SCALE

        if i == 0:
            ref = void_flux[20:150].sum()
            a   = pred[20:150].sum() / ref if ref > 0 else 1.0
            cumulative_scale = a
            print(f'    layer {i}: {mat} {thick}cm  density={density}  a={a:.4f}')
        elif i < len(layers) - 1:
            denom  = np.where(prev_pred_raw[20:150] > 0, prev_pred_raw[20:150], np.nan)
            b_vals = (pred * cumulative_scale)[20:150] / denom
            b = float(np.nanmean(b_vals))
            cumulative_scale *= b

        if i > 0:
            print(f'    layer {i}: {mat} {thick}cm  density={density}')

        prev_pred_raw = pred.copy()

        if i < len(layers) - 1:
            s = pred.sum()
            spec = pred / s if s > 0 else pred

    return pred * cumulative_scale

# ── Load models & base spectrum ──────────────────────────────────────────────
A_SPEC = load_spectrum(spectrum_src)
print(f'Input spectrum: {spectrum_src}  (sum={A_SPEC.sum():.4f})')

VOID_FLUX, _ = load_phits(void_src)
if VOID_FLUX is None:
    raise SystemExit(f'Could not parse void PHITS file: {void_src}')
print(f'Void flux:      {void_src}  (sum={VOID_FLUX.sum():.4e}, bins20:150={VOID_FLUX[20:150].sum():.4e})')

loaded = []
for lbl, folder, scale in model_defs:
    pkl_path = os.path.join(folder, MODEL_PKL_NAME)
    if not os.path.exists(pkl_path):
        print(f'  Skipping {lbl} — pkl not found at {pkl_path}')
        continue
    print(f'Loading: {lbl}  (scale={scale})')
    loaded.append((lbl, load_model(pkl_path), scale))

if not loaded:
    raise SystemExit('No models loaded — check --models paths.')

# ── Transfer head: load if requested ─────────────────────────────────────────
transfer_layer2 = None
if args.transfer:
    transfer_pkl = _K100S2_DIR / args.transfer / 'transfer_model.pkl'
    if not transfer_pkl.exists():
        raise SystemExit(f'Transfer model not found: {transfer_pkl}\n'
                         f'Run scripts/train_transfer_head.py first.')
    print(f'Transfer head ON — model: {transfer_pkl}')
    transfer_layer2 = load_model(str(transfer_pkl))

# ── Bi-modeling: load layer-2 model if requested ──────────────────────────────
bimodel_layer2 = None
if args.bimodel is not None:
    src = args.bimodel
    if src not in _BIMODEL_SOURCES:
        raise SystemExit(f'--bimodel source must be one of {_BIMODEL_SOURCES}, got: {src!r}')
    bimodel_pkl = _K100S2_DIR / f'v_{src}' / MODEL_PKL_NAME
    if not bimodel_pkl.exists():
        raise SystemExit(f'Bi-model pkl not found: {bimodel_pkl}')
    print(f'Bi-modeling ON — layer-2 model: v_{src}  ({bimodel_pkl})')
    bimodel_layer2 = load_model(str(bimodel_pkl))

# ── Build PDF ────────────────────────────────────────────────────────────────
print(f'\nWriting → {out_pdf}')
with PdfPages(out_pdf) as pdf:
    for case_path in phits_files:
        if not os.path.exists(case_path):
            print(f'  Skipping — not found: {case_path}')
            continue

        layers    = parse_layers(case_path)
        case_lbl  = os.path.basename(case_path).replace('.out', '').replace('_', ' ')
        layer_str = ' → '.join(f'{m} {t} cm' for m, t in layers)

        if not layers:
            print(f'  Could not parse layers from filename: {case_path}')
            continue

        unknown = [m for m, _ in layers if m not in MATERIALS]
        if unknown:
            print(f'  Unknown material(s): {unknown}  — skipping {case_lbl}')
            continue

        phits_flux, phits_rerr = load_phits(case_path)
        if phits_flux is None:
            print(f'  Parse failed: {case_path}')
            continue

        phits_dose = dose_excl_bin0(phits_flux)
        phits_sig  = phits_flux * np.nan_to_num(phits_rerr)

        # ── Figure ─────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 11))
        fig.patch.set_facecolor('#f8f9fa')
        gs = gridspec.GridSpec(3, 1, height_ratios=[2.8, 1.2, 1.0],
                               hspace=0.45, figure=fig)
        ax_flux = fig.add_subplot(gs[0])
        ax_rat  = fig.add_subplot(gs[1])
        ax_dose = fig.add_subplot(gs[2])

        PLOT_SLICE = slice(1, 248)
        E_PLOT = E_ALL[PLOT_SLICE]

        ax_flux.semilogy(E_PLOT, phits_flux[PLOT_SLICE], color='black', linewidth=0.8,
                         label='PHITS', zorder=5)
        ax_flux.fill_between(E_PLOT,
                             np.clip(phits_flux[PLOT_SLICE] - phits_sig[PLOT_SLICE], 1e-30, None),
                             phits_flux[PLOT_SLICE] + phits_sig[PLOT_SLICE],
                             color='black', alpha=0.10)

        # Build the list of (label, model, scale, layer2_model) runs for this page.
        # When bi-modeling is on, include the plain baseline run for comparison.
        runs = []
        for lbl, model, scale in loaded:
            if transfer_layer2 is not None:
                runs.append((f'{lbl} [baseline]',  model, scale, None))
                runs.append((f'{lbl} [transfer]',  model, scale, transfer_layer2))
            elif bimodel_layer2 is not None:
                runs.append((f'{lbl} [baseline]',  model, scale, None))
                runs.append((f'{lbl} [bi:{args.bimodel}]', model, scale, bimodel_layer2))
            else:
                runs.append((lbl, model, scale, None))

        dose_rows = []
        preds = {}
        corrected_preds = {}
        for (plot_lbl, model, scale, l2_model), color in zip(runs, MODEL_COLORS):
            pred = multilayer_cnn(model, layers, scale, A_SPEC, VOID_FLUX,
                                  model_layer2=l2_model)
            preds[lbl] = pred

            cnn_plot = pred[PLOT_SLICE]
            mask = (cnn_plot > 0) & np.isfinite(cnn_plot) & (phits_flux[PLOT_SLICE] > 0)
            ratio_vals = np.where(mask, cnn_plot / phits_flux[PLOT_SLICE], np.nan)
            avg_ratio = float(np.nanmean(ratio_vals))
            corrected = pred / avg_ratio
            corrected_preds[plot_lbl] = (corrected, avg_ratio)
            preds[plot_lbl] = pred

            ax_flux.semilogy(E_PLOT, cnn_plot, color=color,
                             linewidth=1.1, linestyle='--', label=plot_lbl)
            ax_flux.semilogy(E_PLOT, corrected[PLOT_SLICE], color=color,
                             linewidth=1.4, linestyle=':', alpha=0.85,
                             label=f'{plot_lbl}  ÷{avg_ratio:.3f} (corrected)')

            d   = dose_excl_bin0(pred)
            pct = 100 * (d - phits_dose) / phits_dose
            d_corr   = dose_excl_bin0(corrected)
            pct_corr = 100 * (d_corr - phits_dose) / phits_dose
            dose_rows.append((plot_lbl, d, pct, d_corr, pct_corr, avg_ratio, color))

        ax_flux.set_xlim(E_PLOT[0], E_PLOT[-1])
        ax_flux.set_ylim(bottom=1e-7)
        ax_flux.set_xlabel('Energy (MeV)', fontsize=11)
        ax_flux.set_ylabel('Neutron flux  [n/cm²/MeV/source]', fontsize=10)
        ax_flux.set_title(f'Flux spectrum: {layer_str}', fontsize=10, fontweight='bold')
        ax_flux.legend(fontsize=8, ncol=2)
        ax_flux.grid(True, which='both', alpha=0.2)
        ax_flux.set_facecolor('#f0f0f0')

        all_ratios = []
        ax_rat.axhline(1, color='black', linewidth=1.0, linestyle='--')
        print(f'\n--- Ratios (PHITS/CNN) for {case_lbl} ---')
        for (lbl, _, _, _, _, avg_ratio, _), color in zip(dose_rows, MODEL_COLORS):

            pred  = preds[lbl]
            corr, _ = corrected_preds[lbl]

            mask  = (pred[PLOT_SLICE] > 0) & np.isfinite(pred[PLOT_SLICE])
            ratio = np.where(mask, phits_flux[PLOT_SLICE] / pred[PLOT_SLICE], np.nan)
            corr_ratio = np.where(mask, phits_flux[PLOT_SLICE] / corr[PLOT_SLICE], np.nan)

            ax_rat.plot(E_PLOT, ratio, color=color, linewidth=0.9, label=lbl)
            ax_rat.plot(E_PLOT, corr_ratio, color=color, linewidth=1.2,
                        linestyle=':', alpha=0.85, label=f'{lbl} corrected')

            valid = ratio[np.isfinite(ratio)]
            all_ratios.append(valid)
            all_ratios.append(corr_ratio[np.isfinite(corr_ratio)])
            print(f'  {lbl}: min={valid.min():.3f}  max={valid.max():.3f}  '
                  f'mean={valid.mean():.3f}  avg_ratio={avg_ratio:.4f}')

        combined = np.concatenate(all_ratios)
        r_min, r_max = combined.min(), combined.max()
        pad = (r_max - r_min) * 0.05
        ax_rat.set_xlim(E_PLOT[0], E_PLOT[-1])
        ax_rat.set_ylim(r_min - pad, r_max + pad)
        ax_rat.set_xlabel('Energy (MeV)', fontsize=11)
        ax_rat.set_ylabel('PHITS / CNN', fontsize=10)
        ax_rat.set_title(f'Ratio  (ylim [{r_min:.2f}, {r_max:.2f}])', fontsize=9)
        ax_rat.legend(fontsize=7)
        ax_rat.grid(True, alpha=0.25)
        ax_rat.set_facecolor('#f0f0f0')

        ax_dose.axis('off')
        col_labels = ['Model', 'avg ratio\n(CNN/PHITS)',
                      'CNN dose\n(pSv/src)', '% err',
                      'Corrected dose\n(pSv/src)', '% err (corr)',
                      'PHITS dose\n(pSv/src)']
        rows = [[lbl, f'{off:.4f}',
                 f'{d:.3e}', f'{pct:+.1f}%',
                 f'{dc:.3e}', f'{pc:+.1f}%',
                 f'{phits_dose:.3e}']
                for lbl, d, pct, dc, pc, off, _ in dose_rows]
        cell_colors = [['white', 'white',
                        'white', _err_color(pct),
                        '#e8f4ff', _err_color(pc),
                        '#d0e8ff']
                       for _, _, pct, _, pc, _, _ in dose_rows]

        tbl = ax_dose.table(cellText=rows, colLabels=col_labels,
                            cellColours=cell_colors,
                            loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#2c3e50')
            tbl[0, j].set_text_props(color='white', fontweight='bold')

        fig.suptitle(
            f'Multilayer Inference — {case_lbl}\n'
            f'Chain: input spec → {layer_str}\n'
            f'PHITS dose (1–250 MeV): {phits_dose:.3e} pSv/source',
            fontsize=11, fontweight='bold', color='#1a252f', y=1.01
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'  Page done: {case_lbl}')

print(f'\nSaved → {out_pdf}')
