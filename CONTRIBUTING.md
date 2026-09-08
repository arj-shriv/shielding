# Contributing

Practical guide for collaborators on this repo: adding reference data, training new
models, and finding your way around the code. For architecture/data-flow details see
[INFERENCE.md](INFERENCE.md); for the full directory layout see [STRUCTURE.md](STRUCTURE.md);
for copy-paste commands see [commands.md](commands.md).

## Setup

```bash
git clone git@github.com:arj-shriv/shielding.git
cd shielding
python3.13 -m venv ../venv          # Keras 3.14.x models need Python 3.13
../venv/bin/pip install -r requirements.txt
../venv/bin/pip install -e ".[dev]"
```

Verify:
```bash
../venv/bin/python -m pytest tests/ -v
../venv/bin/python scripts/utilities/validate_registry.py
```

Run the apps:
```bash
../venv/bin/streamlit run app/main_tool.py
```
**Main Tool** (public) does single/double-layer inference with the primary models.
**Model Explorer** (password-gated, appears in the left nav) documents every model in the
registry and lets you run any model / any multilayer variant, side by side. See
`.streamlit/secrets.toml.example` to set `EXPLORER_PASSWORD` locally.

### Model Explorer's write actions (upload / train / add source — no git needed)

Three tabs in Model Explorer do the same three things this guide walks through below,
from inside the browser: **📤 Upload PHITS**, **🧬 Add source spectrum**, **🏋️ Train model**.
A **📄 Source code** tab shows the actual `.py` files read-only, no gate needed.

These need `GITHUB_PAT` set in `.streamlit/secrets.toml` (a fine-grained, repo-scoped
token — Contents + Pull requests, read/write — see the comments in
`.streamlit/secrets.toml.example`). Without it, those three tabs show as read-only with
a pointer back to this doc. Every write action opens a **pull request** — nothing is
pushed straight to `main` — so use whichever path you prefer: the tabs below, or the
CLI/git steps in this guide, land in the same place.

---

## Adding a new PHITS reference file

1. Drop the `.out` file into the right folder:
   - Single-layer: `data/raw/phits/high_energy/`
   - Multilayer: `data/raw/phits/multilayer/`
2. Name it so the app's matcher (`app/components/phits_matcher.py`) can find it:
   - Single-layer: `<Material>_<thickness>cm.out` (e.g. `Concrete_45cm.out`)
   - Multilayer: `<Mat1>_<t1>cm_<Mat2>_<t2>cm.out` (e.g. `Concrete_45cm_Steel_33cm.out`)
3. Sanity-check it locally before opening a PR:
   ```bash
   ../venv/bin/python scripts/utilities/check_phits_upload.py data/raw/phits/high_energy/Concrete_45cm.out
   ```
4. `git add`, commit, push, open a PR. `.github/workflows/phits_check.yml` runs the same
   check automatically on any PR touching `data/raw/phits/**` (naming convention +
   confirms it parses to exactly 250 flux bins) — it has to pass before merging.
5. Once merged, both apps pick it up automatically (no redeploy) — the PHITS matcher
   re-scans the folder at app startup.

---

## Training a new model

Both training scripts write to `models/registry.json` — the single source of truth both
apps read from — so a model you train shows up in the app the next time it restarts, with
no other wiring needed.

```bash
# k100s2 single-layer variant with a chosen source spectrum
../venv/bin/python scripts/training/train_k100s2.py --name my_model --source concrete25 --epochs 25

# Transfer head (multilayer layer-2) on the frozen k100s2_v1 backbone
../venv/bin/python scripts/training/train_transfer.py --name my_head --arch mlp --epochs 15
```

- `--name` must be unique across the whole registry (primary + experimental + archived) —
  reusing an existing name fails fast with a clear error unless you pass `--overwrite`.
  This is deliberate: it stops an accidental retrain from silently clobbering a model
  another script or teammate is relying on.
- New models land under `models/.../experimental/<name>/` by default. Pass `--primary`
  only if this is meant to replace the model the Main Tool uses by default — do that
  deliberately, not as the default path.
- Each run also writes `models/.../<name>/description.txt` — edit it after training if
  the auto-generated one-liner needs more context for the next person.
- Training data comes from `data/raw/response_matrices/{Concrete,Steel,BPE}/` — already
  in the repo, no external data directory needed.
- `scripts/inference/single_layer.py` and `scripts/inference/multilayer.py` are the
  importable functions both apps and the training scripts call — read those first if
  you want to understand exactly what a forward pass does (input tensor layout, output
  scale, dose conversion).

After training, run the registry validator once before pushing:
```bash
../venv/bin/python scripts/utilities/validate_registry.py
```
CI (`.github/workflows/ci.yml`) runs the same check plus the test suite on every push/PR.

---

## Adding a new source spectrum

Source spectra live in `src/shielding_ml/data/source_spectra.py` and are what
`train_k100s2.py --source <name>` trains against. Adding one is two steps:

1. **Fit it from an existing PHITS file** (needs `data/raw/phits/high_energy/` to already
   have the reference — add that first if it doesn't, see above):
   ```python
   # in src/shielding_ml/data/source_spectra.py
   concrete100 = _make_noisy_source('Concrete_100cm.out', E_cut=10.0)
   ```
   `_make_noisy_source` auto-fits the piecewise low/high-energy model plus a log-normal
   noise envelope from the residuals — you don't need to hand-fit anything. `E_cut` (MeV)
   is the boundary between the low- and high-energy fit regimes; 10.0 works for every
   existing source and is a reasonable default.

2. **Register it** so `--source concrete100` resolves:
   ```python
   SOURCE_FUNCTIONS: dict[str, object] = {
       'tracknet10': tracknet10,
       'concrete25': concrete25,
       'concrete45': concrete45,
       'steel27':    steel27,
       'bpe25':      bpe25,
       'concrete100': concrete100,   # <- add here
   }
   ```

Then train against it exactly like any other source:
```bash
../venv/bin/python scripts/training/train_k100s2.py --name v_concrete100 --source concrete100
```

If the source isn't a simple curve fit to one PHITS file (e.g. a hand-specified function
like `tracknet10`, which is hardcoded piecewise6 params rather than fit from data — see
the docstring at the top of `source_spectra.py`), write it as a plain
`f(E_MeV: ndarray[250]) -> ndarray[250]` callable and register it the same way.

---

## Where the code actually lives

| I want to... | Look at |
|---|---|
| See what a forward pass does | `scripts/inference/single_layer.py`, `scripts/inference/multilayer.py` |
| See how multilayer chaining variants differ | `scripts/inference/multilayer.py` — each variant (A/B/C) is a separate, documented function |
| Change what the Main Tool shows | `app/main_tool.py` + `app/components/` |
| Change what the Model Explorer shows | `app/pages/2_Model_Explorer.py` |
| Train a model | `scripts/training/train_k100s2.py`, `scripts/training/train_transfer.py` |
| See/edit registry read-write logic | `scripts/training/registry_utils.py` |
| Change dose/flux math | `src/shielding_ml/metrics/dose.py`, `src/shielding_ml/data/loaders.py` |

## Before opening a PR

```bash
../venv/bin/python -m pytest tests/ -v
../venv/bin/python scripts/utilities/validate_registry.py
../venv/bin/python scripts/utilities/check_phits_upload.py --all   # if you touched PHITS data
```
CI runs all of this automatically, but catching it locally first is faster than waiting on
a red check.
