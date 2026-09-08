# shielding-ml

A 1D CNN surrogate model for neutron shielding dose prediction — trained to replace
PHITS Monte Carlo transport simulations for a Ca-48 @ 150 MeV/n beam through Concrete,
Steel, and BPE shielding, single-layer or chained multilayer.

Given a shield's material(s) and thickness(es), the model predicts the transmitted
neutron flux spectrum (250 bins, 0–250 MeV) and the resulting effective dose rate, in
milliseconds instead of the hours a PHITS run takes.

## Quick start

```bash
git clone git@github.com:arj-shriv/shielding.git
cd shielding
python3.13 -m venv ../venv
../venv/bin/pip install -r requirements.txt
../venv/bin/pip install -e ".[dev]"
../venv/bin/streamlit run app/main_tool.py
```

That launches the **Main Tool**: pick single- or double-layer, a material and thickness,
and the flux spectrum + dose rate update live. If a PHITS reference exists for that exact
geometry it's overlaid automatically with the dose error shown.

A second page, **Model Explorer** (left nav, password-gated), documents every model in
the registry and lets you run any model or multilayer chaining variant, including two
side by side.

Prefer a notebook? `inference_notebook.ipynb` covers the same ground for local,
non-interactive use.

## The two production models

- **`k100s2_v1`** — single-layer CNN (474K params, ~9× fewer than the original TrackNet10
  architecture it replaced, with equal or better accuracy)
- **`transfer_mlp`** — multilayer layer-2 head: the entire `k100s2_v1` backbone frozen,
  with a small (32K param) trainable correction head on top, fed the original TrackNet10
  beam as its source spectrum (not the layer-1 output — that path was tried and performs
  worse; see `scripts/inference/multilayer.py` for why)

## Repo layout

```
app/                    Streamlit apps (Main Tool + Model Explorer)
scripts/inference/      Importable prediction functions — single-layer, multilayer (all variants)
scripts/training/       Training scripts that register results in models/registry.json
scripts/analysis/       Uncertainty quantification, dose error tables
src/shielding_ml/       Installable library: data loaders, dose math, source spectra
models/registry.json    Every trained model — architecture, params, val MAE, description
data/raw/phits/         PHITS reference files (what the app overlays for comparison)
data/raw/response_matrices/   Training data — response matrices per material/thickness
```

Full breakdown in [STRUCTURE.md](STRUCTURE.md).

## Contributing

Adding a PHITS reference file, training a new model, or adding a new source spectrum to
train against are all meant to be a few lines of code or a drag-and-drop — see
[CONTRIBUTING.md](CONTRIBUTING.md) for exact steps. [commands.md](commands.md) has the
copy-paste command reference; [INFERENCE.md](INFERENCE.md) covers the input/output tensor
layout and dose-computation math in detail.

CI (`.github/workflows/`) validates the model registry and PHITS file naming/format on
every push and PR.
