# Commands

All commands run from the repo root: `/Users/arjun/Python_Codes_Refactored/shielding-ml`  
Always use `../venv/bin/python`, not system Python.

---

## Training

### Source-matched k100s2 models
Train the k100s2 architecture with a specific shielded source spectrum.  
Saves to `/Users/arjun/Python_Codes/<YYYYMMDD>/k100s2_<source>_src/`

```bash
# Single source
../venv/bin/python scripts/train_k100s2_multisrc.py --source concrete25
../venv/bin/python scripts/train_k100s2_multisrc.py --source steel27
../venv/bin/python scripts/train_k100s2_multisrc.py --source bpe25
../venv/bin/python scripts/train_k100s2_multisrc.py --source tracknet10  # baseline

# All three shielded sources sequentially
EPOCHS=25 bash scripts/train_all_sources.sh
```

Available sources: `concrete25`, `steel27`, `bpe25`, `tracknet10`  
Key flags: `--epochs 25`, `--loss mae`

---

### Transfer-learned correction head
Freezes entire k100s2 v1 model, trains a new head on top. Three variants:

```bash
# Linear head (Dense 250) — concrete25 source
../venv/bin/python scripts/train_transfer_head.py --arch linear --sources concrete25 --epochs 10

# MLP head (Dense 64 → Dense 250) — piecewise concrete45 source
../venv/bin/python scripts/train_transfer_head.py --arch mlp --sources concrete45 --epochs 30

# MLP head — actual CNN layer-1 predictions as source (best val MAE: 0.077)
../venv/bin/python scripts/train_transfer_cnn_source.py
../venv/bin/python scripts/train_transfer_cnn_source.py --epochs 25 --n_per_src 10000

# Conv1D head (Conv50×50 → Conv10×10 → Dense 250) — CNN layer-1 sources (val MAE: 0.046)
../venv/bin/python scripts/train_transfer_conv_head.py
../venv/bin/python scripts/train_transfer_conv_head.py --epochs 25
```

Key flags: `--sources`, `--epochs`, `--loss mae`, `--n_per_src`

---

## Inference

### Interactive notebook
```
inference_notebook.ipynb
```
Open with the `../venv/bin/python` kernel (Python 3.13). Covers single-layer inference, multilayer primary mode (TrackNet10 source + transfer_mlp), and loading experimental models. Best starting point for a new case.

---

### Single-layer + multilayer (all registered models)
Runs every model in `models/` against all single-layer and multilayer PHITS cases.  
Saves to `inference_results/<model-id>/single_layer/inference.pdf` and `.../multilayer/inference.pdf`

```bash
../venv/bin/python scripts/run_all_inference.py
../venv/bin/python scripts/run_all_inference.py --multilayer-only
../venv/bin/python scripts/run_all_inference.py --model-id k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240
```

---

### Multilayer inference (flexible)
Runs chained multilayer inference. Auto-discovers all 5 compound PHITS cases.

**Standard (k100s2 v1, TrackNet10-trained):**
```bash
../venv/bin/python scripts/inference/run_multilayer.py
```
Output: `inference_results/k100s2-.../multilayer/multilayer_comparison.pdf`

---

**Bi-model (layer 1 = v1 TrackNet10, layer 2 = source-matched k100s2):**
```bash
../venv/bin/python scripts/inference/run_multilayer.py --bimodel
../venv/bin/python scripts/inference/run_multilayer.py --bimodel concrete25
../venv/bin/python scripts/inference/run_multilayer.py --bimodel steel27
```
Output: `inference_results/bi_model_multilayer/multilayer_comparison.pdf`

---

**Transfer-learned head (layer 1 = v1 frozen, layer 2 = transfer head):**
```bash
../venv/bin/python scripts/inference/run_multilayer.py --transfer
```
Output: `inference_results/transfer_learned/multilayer_comparison.pdf`  
Requires: `models/k100s2-.../transfer_learned/transfer_model.pkl` (run training first)

---

**Custom model or PHITS files:**
```bash
../venv/bin/python scripts/inference/run_multilayer.py \
    --layer1 models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/v1:0.001 \
    --phits data/raw/phits/multilayer/Concrete_45cm_Steel_33cm.out \
    --out my_output.pdf
```

---

**TrackNet10 as layer-2 source — primary/recommended mode:**
```bash
../venv/bin/python scripts/inference/run_tracknet_source_l2.py
```
Uses `primary/transfer_mlp` for layer 2, fed with the original TrackNet10 source (not the normalised CNN layer-1 output). Layer-1 amplitude attenuation (`a`-factor) is extracted from `primary/k100s2_v1` and applied to the final output. Auto-discovers all `.out` files in `data/raw/phits/multilayer/`.  
Output: `inference_results/tracknet_source_l2/comparison.pdf`  
Results: dose error −28% to +18% across all 6 multilayer cases.

---

## Model locations

All models live under: `models/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/`

### Primary models (`primary/`)

| Model | Path | pkl file |
|---|---|---|
| k100s2 v1 — single-layer backbone | `primary/k100s2_v1/` | `model.pkl` |
| Transfer MLP — multilayer layer-2 head | `primary/transfer_mlp/` | `transfer_model.pkl` |

Backward-compat symlinks in the model-id root: `v1 → primary/k100s2_v1`, `transfer_cnn_source_mlp → primary/transfer_mlp`. All existing scripts continue to work unchanged.

### Experimental models (`experimental/`)

| Model | Subfolder | pkl file |
|---|---|---|
| k100s2 concrete25 source | `v_concrete25/` | `model.pkl` |
| k100s2 steel27 source | `v_steel27/` | `model.pkl` |
| k100s2 bpe25 source | `v_bpe25/` | `model.pkl` |
| Transfer — linear head (concrete25) | `transfer_learned/` | `transfer_model.pkl` |
| Transfer — MLP head (concrete45 piecewise) | `transfer_concrete45_mlp/` | `transfer_model.pkl` |
| Transfer — Conv1D head (CNN layer-1 sources) | `transfer_conv_head/` | `transfer_model.pkl` |

---

## Analysis & Uncertainty

### Density uncertainty — single-layer
Varies density ±10% around nominal (50 points, 25 each direction) for each (material, thickness) and records CNN-predicted dose. Materials: Concrete/Steel/BPE. Thicknesses: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140 cm.

```bash
../venv/bin/python scripts/analysis/density_uncertainty.py
../venv/bin/python scripts/analysis/density_uncertainty.py --plots-only   # replot from CSV
../venv/bin/python scripts/analysis/density_uncertainty.py --pct-range 5  # ±5% instead
```

Outputs → `inference_results/density_uncertainty/`:
- `results.csv` — 1800 rows (material, thickness, nominal density, density, offset %, dose)
- `scatter.pdf` — dose vs density offset % and vs absolute density (log-scale, colour = thickness)
- `histograms.pdf` — dose histogram + Gaussian fit per (material, thickness), 3 pages

---

### Thickness uncertainty — single-layer
Samples 1000 thicknesses from N(μ=nominal, σ=1 cm) for each (material, nominal thickness) and records CNN-predicted dose. Nominal thicknesses: 25, 60, 90 cm.

```bash
../venv/bin/python scripts/analysis/thickness_uncertainty.py
../venv/bin/python scripts/analysis/thickness_uncertainty.py --sigma 2     # wider distribution
../venv/bin/python scripts/analysis/thickness_uncertainty.py --plots-only  # replot from CSV
```

Outputs → `inference_results/thickness_uncertainty/`:
- `results.csv` — 9000 rows
- `distributions.pdf` — Page 1: 3×4 grid (col 0 = input thickness Gaussian, cols 1–3 = dose histogram + Gaussian fit per material); Page 2: dose vs sampled thickness scatter; Page 3: summary table (μ, σ, CV% per case)

---

### Dose error table — old vs new model
Compares old TrackNet10 model vs new k100s2 (v1) dose % error against all single-layer PHITS reference cases (same discovery as `run_all_inference.py`).

```bash
../venv/bin/python scripts/analysis/dose_error_table.py
```

Output → `inference_results/dose_error_comparison/dose_error_table.pdf`  
Colour coding: green < 20%, yellow 20–30%, red > 30% absolute error.  
Average absolute errors shown below the table.

---

## Multilayer experiments log

All 5 PHITS multilayer cases: Concrete 45+Steel 33, Concrete 67+Steel 35, Concrete 73+BPE 27, Steel 23+Concrete 65, Steel 35+Concrete 94.  
Performance metric: `avg_ratio = mean(CNN / PHITS)` over bins 20–150. Ideal = 1.0. Concrete-first cases overpredicted (ratio 3–7×); Steel-first cases underpredicted (ratio 0.3–0.4×).

---

### 1. Baseline multilayer (v1 model, TrackNet10 source)
- **What**: Chain k100s2-v1 twice. Layer-1 source = TrackNet10 (normalized). Layer-2 source = normalized CNN layer-1 output. Amplitude carried via `a = CNN_L1_flux[20:150] / void_flux[20:150]`.
- **Result**: avg_ratio ≈ 3.3–6.4 for concrete-first, 0.31–0.40 for steel-first. Systematic 3–6× amplitude error.
- **Files**:
  - Script: `scripts/inference/run_multilayer.py`
  - Output: `inference_results/k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/multilayer/multilayer_comparison.pdf`
  - Model: `models/k100s2-.../v1/model.pkl`

---

### 2. Bi-model (source-matched layer-2 model)
- **What**: Layer 1 = v1 (TrackNet10). Layer 2 = separate k100s2 model trained from scratch using a shielded source spectrum (concrete25, steel27, or bpe25) as the training source, so its input distribution better matches the shielded layer-1 output.
- **Result**: avg_ratio nearly identical to baseline. No measurable improvement across all 5 cases.
- **Why it failed**: The normalized CNN layer-1 output does not look like the concrete25/steel27 piecewise source functions — the distribution mismatch persists even with a source-matched model.
- **Files**:
  - Script: `scripts/inference/run_multilayer.py --bimodel [concrete25|steel27|bpe25]`
  - Output: `inference_results/bi_model_multilayer/multilayer_comparison.pdf`
  - Models: `models/k100s2-.../v_concrete25/`, `v_steel27/`, `v_bpe25/`

---

### 3. Transfer head — linear (frozen v1 + Dense 250)
- **What**: Freeze all v1 weights. Add a single new Dense(250) layer after the frozen Dense(250) output. Train only the 250×250+250 new parameters on piecewise concrete25 source spectra via response-matrix convolution. Used for layer 2 only; layer 1 still uses frozen v1 directly.
- **Result**: Validation MAE plateaued at ~0.173 (same as baseline). No improvement in inference.
- **Why it failed**: Capacity too low; the linear head cannot correct for non-linear spectral shape distortions.
- **Files**:
  - Training script: `scripts/train_transfer_head.py --arch linear --sources concrete25`
  - Model: `models/k100s2-.../transfer_concrete45_linear/transfer_model.pkl`
  - Inference: `scripts/inference/run_multilayer.py --transfer transfer_concrete45_linear`

---

### 4. Transfer head — MLP (frozen v1 + Dense 64 → Dense 250), piecewise concrete45 source
- **What**: Same frozen-backbone approach, but with a two-layer MLP head (Dense 64 ReLU → Dense 250). Trained using a piecewise fit to the Concrete 45cm PHITS output as the source spectrum (concrete45, added to source_spectra.py). 32K learnable parameters.
- **Result**: Validation MAE reached ~0.150 (improved from 0.173). Still worsened all 5 inference cases vs baseline.
- **Why it failed**: Head trained on piecewise-approximated source spectra; at inference it sees the actual CNN layer-1 output — a different distribution.
- **Files**:
  - Training script: `scripts/train_transfer_head.py --arch mlp --sources concrete45 --epochs 30`
  - Source fit: `src/shielding_ml/data/source_spectra.py` (concrete45 entry)
  - Model: `models/k100s2-.../transfer_concrete45_mlp/transfer_model.pkl`
  - Inference: `scripts/inference/run_multilayer.py --transfer transfer_concrete45_mlp`
  - Output: `inference_results/transfer_concrete45_mlp/multilayer_comparison.pdf`

---

### 7. Transfer head — Conv1D, trained on actual CNN layer-1 predictions as source
- **What**: Same backbone cut as #5 but replaces the MLP head with two new Conv1D layers on top of the frozen Conv1D_3 output (16, 64). New head: Conv1D(50, k=50, s=1, same) → Conv1D(10, k=10, s=2, same) → Flatten(80) → Dense(250). Training data identical to #5.
- **New trainable params**: 185,310 (vs 20,410 for MLP head).
- **Files**:
  - Training script: `scripts/train_transfer_conv_head.py`
  - Model: `models/k100s2-.../transfer_conv_head/transfer_model.pkl`

```bash
../venv/bin/python scripts/train_transfer_conv_head.py
../venv/bin/python scripts/train_transfer_conv_head.py --epochs 25
```

---

### 5. Transfer head — MLP, trained on actual CNN layer-1 predictions as source
- **What**: Same MLP head architecture. Instead of piecewise fits, generates training sources by actually running v1 CNN on Concrete/Steel/BPE at 5 thicknesses each (15 source spectra) using TrackNet10 input. These normalized CNN outputs are used as layer-2 source spectra, with layer-2 targets computed via response-matrix convolution. 10K samples/material (2K/thickness), 30K total.
- **Result**: Validation MAE reached 0.077 (much better). Still worsened all 5 inference cases (avg_ratio 3.5–7.2 vs baseline 3.3–6.5).
- **Why it failed**: The amplitude error is embedded in the `a` factor from the layer-1 CNN prediction. The transfer head output is multiplied by `a` after the fact; if `a` is wrong (CNN under/over-attenuates layer 1), the final output inherits that error regardless of how well the head learned the spectral shape.
- **Files**:
  - Training script: `scripts/train_transfer_cnn_source.py`
  - Model: `models/k100s2-.../transfer_cnn_source_mlp/transfer_model.pkl`
  - Inference: `scripts/inference/run_multilayer.py --transfer transfer_cnn_source_mlp`
  - Output: `inference_results/transfer_cnn_source_mlp/multilayer_comparison.pdf`

---

### 8. Source variant diagnostic (Concrete 45 + Steel 33 only)
- **What**: Three-way comparison to isolate where the error comes from:
  - **(B)** Use actual PHITS Concrete 45cm output (not CNN) as the layer-2 source. This bypasses layer-1 CNN error entirely.
  - **(C)** Bi-model with v1 for layer 1 and v_concrete25 for layer 2.
- **Result**:
  - Baseline avg_ratio = 3.22.
  - **(B)** avg_ratio = 8.81 — dramatically worse even with the true intermediate spectrum.
  - **(C)** avg_ratio = 3.15 — negligible improvement.
- **Key finding**: Using the real PHITS intermediate spectrum as layer-2 source makes the CNN predict 2.2× higher raw flux (2.89 vs 1.33 sum in bins 20–150). The normalized PHITS Concrete 45 spectrum is out-of-distribution for a model trained on TrackNet10 sparse random sources. The CNN layer-1 output, even though amplitude-wrong, happens to have a spectral shape closer to the training distribution, causing less damage in layer 2.
- **Conclusion**: The amplitude error is not the only problem. The model is fundamentally out-of-distribution for any real shielded spectrum input. The fix would require training the model end-to-end on multilayer examples (source→layer1→layer2 with PHITS multilayer targets), not patching a frozen single-layer model.
- **Files**:
  - Script: `scripts/inference/compare_source_variants.py`
  - Output: `inference_results/source_variant_comparison.png`
