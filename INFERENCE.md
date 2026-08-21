# Inference & Dose Computation

## Overview

The CNN takes a neutron **input spectrum + shielding geometry** and predicts the **transmitted flux spectrum** (250 bins, 0–250 MeV). Dose is then computed from that predicted flux using the ICRP fluence-to-effective-dose conversion table.

---

## Where things live

| Thing | Path |
|-------|------|
| Primary models | `models/<model-id>/primary/` |
| k100s2 v1 (single-layer backbone) | `models/<model-id>/primary/k100s2_v1/model.pkl` |
| Transfer MLP (multilayer layer-2 head) | `models/<model-id>/primary/transfer_mlp/transfer_model.pkl` |
| Experimental / deprecated variants | `models/<model-id>/experimental/<variant>/` |
| Backward-compat symlinks | `v1 → primary/k100s2_v1`, `transfer_cnn_source_mlp → primary/transfer_mlp` |
| Model registry (all models + status) | `models/registry.csv` |
| PHITS single-layer references | `data/raw/phits/high_energy/` |
| PHITS multilayer references | `data/raw/phits/multilayer/` |
| Input beam spectrum | `data/raw/spectra/TrackNet10_spectrum.dat` |
| ICRP dose conversion table | `data/reference/Energy_to_Effective_Dose.txt` |
| Unshielded reference flux | `data/reference/No_Shielding.out` |
| Interactive inference notebook | `inference_notebook.ipynb` |

---

## Single-layer inference

**Input to CNN** (shape `[1, 250, 3]`):
```
channel 0: normalised input spectrum  (TrackNet10_spectrum.dat, sums to ~1)
channel 1: thickness  × input_scale   (e.g. 25 × 0.001 = 0.025)
channel 2: density    × input_scale   (e.g. 2.3 × 0.001 = 0.0023)
```

**Densities**: Concrete = 2.3 g/cm³, Steel = 7.86 g/cm³, BPE = 1.04 g/cm³

**Output**: `log10(flux)` for 250 bins → `flux = 10^pred × FLUX_SCALE`  
**FLUX_SCALE** = 200 × 200 × 50 = 2,000,000

**Dose** (mrem/h):
```
dose = sum(flux[1:250] × R[1:250])   # excludes bin 0 (0–1 MeV)
```
where `R[i]` = ICRP effective dose coefficient interpolated at bin midpoint (0.5, 1.5, …, 249.5 MeV).

---

## Multilayer (chained) inference

**Primary mode** — uses `k100s2_v1` for layer 1 and `transfer_mlp` for layer 2 with the original TrackNet10 source. This keeps layer 2 in-distribution and gives dose error −28% to +18% across all 6 multilayer cases.

For a compound shield `Mat1 thick1 → Mat2 thick2`:

1. **Layer 1**: Run `k100s2_v1` with `(TrackNet10_source, mat1, thick1)` → `pred_L1`
2. **Attenuation factor**: `a = sum(pred_L1[20:150]) / sum(void_flux[20:150])`  
   where `void_flux` comes from `No_Shielding.out`; captures layer-1 amplitude attenuation
3. **Layer 2**: Run `transfer_mlp` with `(TrackNet10_source, mat2, thick2)` → `pred_L2`  
   Note: uses the **original TrackNet10 source**, not the CNN layer-1 output — keeps the model in-distribution
4. **Final flux**: `pred_final = pred_L2 × a`

The factor `a` carries the layer-1 amplitude information so that the final output reflects the overall beam attenuation, even though `transfer_mlp` was fed the unnormalized TrackNet10 source in step 3.

---

## Running inference

### Interactive notebook (recommended starting point)
```
inference_notebook.ipynb
```
Open with the `../venv/bin/python` kernel. Covers single-layer, multilayer (primary mode), and loading experimental models. Cells are commented with explanations.

### Primary multilayer script (TrackNet10 source, transfer_mlp)
```bash
../venv/bin/python scripts/inference/run_tracknet_source_l2.py
```
Output: `inference_results/tracknet_source_l2/comparison.pdf`

### Single report scripts (pre-configured model subsets)
```bash
../venv/bin/python scripts/reports/make_result_content.py      # TrackNet10 vs New CNN, single-layer
../venv/bin/python scripts/reports/make_dose_pdf.py            # All 8 original models, full dose table
../venv/bin/python scripts/reports/dose_vs_thickness.py        # 2 models, dose vs thickness curve
../venv/bin/python scripts/reports/dose_by_model.py            # 8 models × 3 materials per page
../venv/bin/python scripts/inference/run_multilayer.py         # Multilayer comparison (all 6 cases)
```

### Multilayer with transfer head (layer 2) — experimental variants
```bash
# Default transfer head (transfer_concrete45_mlp):
../venv/bin/python scripts/inference/run_multilayer.py --transfer

# Specific transfer head:
../venv/bin/python scripts/inference/run_multilayer.py --transfer transfer_cnn_source_mlp
../venv/bin/python scripts/inference/run_multilayer.py --transfer transfer_conv_head
```

### Adding new PHITS data
- Single-layer: drop `.out` files into `data/raw/phits/high_energy/`
- Multilayer: drop `.out` files into `data/raw/phits/multilayer/`  
  Name format: `<Mat1>_<X>cm_<Mat2>_<Y>cm.out` (auto-parsed by the inference pipeline)

### Adding a new model
1. Copy `model.pkl` to `models/<new-id>/v1/`
2. Add `config.yaml` and `metadata.json` alongside it
3. Add a row to `models/registry.csv`
4. Re-run `python scripts/run_all_inference.py`

---

## Metrics computed

| Metric | Formula | Where shown |
|--------|---------|-------------|
| Flux MAPE | `mean(|CNN - PHITS| / PHITS)` over bins where PHITS > 0 | flux plot annotation |
| Flux MSE | `mean((CNN - PHITS)²)` over bins 1–249 | flux plot annotation |
| Dose % error | `100 × (CNN_dose - PHITS_dose) / PHITS_dose` | dose table |
| avg_ratio | `mean(CNN / PHITS)` over bins 20–150 where PHITS > 0 | multilayer comparison |

---

## Uncertainty analysis

### Density uncertainty
Script: `scripts/analysis/density_uncertainty.py`  
Varies density ±10% (50 samples) at fixed thickness. Plots dose histogram + Gaussian fit per (material, thickness).  
Output: `inference_results/density_uncertainty/`

### Thickness uncertainty
Script: `scripts/analysis/thickness_uncertainty.py`  
Samples thickness from N(μ, σ=1 cm) for nominal thicknesses 25, 60, 90 cm.  
Shows input Gaussian alongside output dose distribution with Gaussian fit and summary table.  
Output: `inference_results/thickness_uncertainty/`

### Dose error comparison (old vs new model)
Script: `scripts/analysis/dose_error_table.py`  
Produces colour-coded table of dose % error for old TrackNet10 model vs new k100s2 v1,  
against all single-layer PHITS reference cases. Average absolute error shown below table.  
Output: `inference_results/dose_error_comparison/dose_error_table.pdf`

**Always use `../venv/bin/python`** — models were saved with Keras 3.14.1 (Python 3.13 venv).  
Base conda Python 3.12 will fail to deserialise them (`quantization_config` mismatch).
