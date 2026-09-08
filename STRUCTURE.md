# Repository Structure

```
shielding-ml/
│
├── inference_notebook.ipynb   # Interactive single-layer + multilayer inference (local use)
│
├── app/                        # Streamlit tool suite
│   ├── main_tool.py            # Main Tool — public-facing single/double layer inference
│   ├── pages/
│   │   └── 2_Model_Explorer.py # Model Explorer — docs + all-variant inference (password-gated)
│   └── components/
│       ├── model_loader.py     # Load models from registry.json (st.cache_resource)
│       ├── phits_matcher.py    # Map (mat, thick[, mat2, thick2]) -> PHITS reference path
│       └── flux_plot.py        # Shared semilogy flux-spectrum plot
│
├── src/shielding_ml/          # Installable Python package (pip install -e .)
│   ├── data/
│   │   ├── loaders.py         # Read PHITS .out files and spectra
│   │   └── constants.py       # Shared constants (MATERIALS, FLUX_SCALE, energy grids)
│   ├── metrics/
│   │   ├── dose.py            # Dose integration (midpoint and geometric-mean R)
│   │   └── chi2.py            # Reduced chi-squared and flux MAPE
│   ├── models/
│   │   └── inference.py       # Load a model.pkl and predict flux
│   ├── pipelines/
│   │   ├── paths.py           # Central path registry (REPO_ROOT, PHITS dirs, etc.)
│   │   └── run_manager.py     # Create dated run dirs, write metadata
│   └── plotting/              # (stub — future plotting helpers)
│
├── scripts/                   # Runnable entrypoints, organised by purpose
│   ├── reports/               # Generate PDFs and comparison figures
│   │   ├── make_result_content.py       → result_content/comparison.pdf
│   │   ├── make_result_dose_by_model.py → result_content/dose_by_model.pdf
│   │   ├── make_dose_pdf.py             → dose_comparison.pdf
│   │   ├── dose_by_model.py             → dose_by_model.pdf
│   │   └── dose_vs_thickness.py         → dose_vs_thickness.pdf
│   ├── inference/
│   │   ├── single_layer.py              # Importable: load_model / predict_flux / compute_dose
│   │   ├── multilayer.py                # Importable: Variants A/B/C — B (TrackNet10 source) is primary
│   │   ├── run_multilayer.py            # Chain CNN across two shielding layers
│   │   └── run_tracknet_source_l2.py    # Primary multilayer mode (TrackNet10 source + transfer_mlp)
│   ├── training/
│   │   ├── registry_utils.py            # Shared registry read/write/exists-check for both scripts below
│   │   ├── train_k100s2.py              # --name --source --primary --overwrite -> registers in registry.json
│   │   └── train_transfer.py            # --name --arch mlp|linear|conv --primary --overwrite
│   ├── analysis/
│   │   ├── density_uncertainty.py       # Dose sensitivity to ±10% density variation
│   │   ├── thickness_uncertainty.py     # Dose sensitivity to N(μ, σ=1 cm) thickness variation
│   │   └── dose_error_table.py          # Old TrackNet10 vs new k100s2 dose % error table
│   ├── evaluate/
│   │   └── geom_response.py             # Compare midpoint vs geometric-mean dose
│   └── utilities/
│       ├── validate_registry.py         # Schema-check models/registry.json (used by CI)
│       ├── check_phits_upload.py        # Sanity-check new PHITS files (naming + 250-bin parse; used by CI)
│       ├── copy_study_files.py          # SLAC cluster file management
│       └── extract_tracknet10.py
│
├── data/
│   ├── raw/
│   │   ├── phits/
│   │   │   ├── high_energy/   # PHITS .out files (Concrete/Steel/BPE, various thicknesses)
│   │   │   ├── low_energy/    # LE PHITS files (1 keV–1 MeV)
│   │   │   └── multilayer/    # Compound-shield PHITS files (Mat1_Xcm_Mat2_Ycm.out)
│   │   ├── spectra/           # Input beam spectra (Ca-48 150 MeV/n, TrackNet10)
│   │   └── response_matrices/ # ICRP fluence-to-dose response (.dat files)
│   └── reference/             # Energy_to_Effective_Dose.txt, No_Shielding.out
│
├── models/
│   ├── registry.csv           # Legacy index (kept for reference)
│   ├── registry.json          # Source of truth — read by both Streamlit apps and both
│   │                          #   training scripts; sections: primary / experimental / archived
│   └── k100s2-k25s2-k11-k3-d256-d64-mae-bins50-240/
│       ├── primary/
│       │   ├── k100s2_v1/     # Main single-layer model (474K params)  → model.pkl
│       │   └── transfer_mlp/  # Multilayer layer-2 head (32K params)   → transfer_model.pkl
│       ├── experimental/      # Comparison variants — new trained models land here by default
│       │   ├── v_concrete25/, v_steel27/, v_bpe25/   # source-matched k100s2 variants
│       │   ├── transfer_learned/, transfer_concrete45_mlp/  # earlier transfer heads
│       │   └── transfer_conv_head/                   # Conv1D transfer head
│       ├── v1 → primary/k100s2_v1               # symlink (backward compat)
│       └── transfer_cnn_source_mlp → primary/transfer_mlp  # symlink (backward compat)
│
├── images_for_paper/          # Publication-quality PNGs (300 DPI)
│   ├── tracknet10_spectrum.png           # TrackNet10 source + piecewise6 fit
│   ├── architecture/                     # k100s2_v1.png, transfer_mlp.png — layer diagrams
│   │                                     #   (regenerated from live .pkl shapes by
│   │                                     #    plot_model_architectures.py, embedded in Main Tool sidebar)
│   ├── single_layer/                     # One PNG per (material, thickness) case
│   └── multilayer/                       # One PNG per compound-shield case
│
├── inference_results/         # Script outputs (auto-generated, not committed)
│   ├── tracknet_source_l2/comparison.pdf
│   ├── density_uncertainty/
│   └── thickness_uncertainty/
│
├── configs/
│   └── paths.yaml             # Override data/model paths without editing code
│
├── .streamlit/
│   ├── config.toml            # App theme
│   └── secrets.toml.example   # Copy to secrets.toml (gitignored) — EXPLORER_PASSWORD etc.
│
├── .github/workflows/
│   ├── ci.yml                 # pytest + registry validation on every push/PR to main
│   └── phits_check.yml        # PR check on data/raw/phits/** changes — naming + 250-bin parse
│
├── tests/
│   ├── test_loaders.py
│   ├── test_registry.py       # validate_registry / registry_utils / check_phits_upload
│   └── test_run_manager.py
│
├── archive/legacy_scripts/    # Old scripts with hardcoded Windows paths (kept for reference)
├── reports/                   # Output PDFs land here when run from this repo
├── runs/                      # Dated inference/evaluation run outputs
├── requirements.txt           # Streamlit app runtime deps (streamlit, keras>=3.14, tensorflow>=2.17)
└── pyproject.toml             # Package metadata and dependencies
```

## How it fits together

- **`src/shielding_ml`** is the shared library. Scripts import from it (`from shielding_ml.metrics.dose import dose_midpoint`).
- **`scripts/`** are thin entrypoints — they parse args, call library functions, and write outputs.
- **`app/`** is the Streamlit tool suite; it imports from `scripts/inference/` and `app/components/`, never duplicates logic.
- **`data/raw/`** holds inputs (including `response_matrices/`, needed by `scripts/training/`); **`models/`** holds trained weights; **`reports/`** and **`runs/`** hold outputs.
- **`models/registry.json`** is the single source of truth for which model is which — both apps and both training scripts read/write it. `registry.csv` is legacy and no longer updated.
- Training a model that reuses an existing `--name` is refused unless `--overwrite` is passed (see `scripts/training/registry_utils.require_available`).

## What still lives in Python_Codes (not yet ported)

`run_inference.py`, `make_chi2_table.py`, `make_model_table.py`, `make_train_pdf.py`,
`plot_adj_ratio.py`, `plot_dose_contributions.py`, `compare_results.py`, `compute_dose_table.py`
