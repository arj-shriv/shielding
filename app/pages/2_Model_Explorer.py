"""
Shielding CNN — Model Explorer
===============================
Collaborator-only page: full model documentation + advanced inference tool
exposing every single-layer model and every multilayer chaining variant.

Password-gated behind EXPLORER_PASSWORD (see .streamlit/secrets.toml.example).
Run via the Main Tool (app/main_tool.py) — this file lives in app/pages/ so
Streamlit's multipage nav picks it up automatically.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shielding_ml.data.constants import MATERIALS, E_ALL
from shielding_ml.data.loaders import load_phits
from shielding_ml.pipelines.paths import PHITS_HE, PHITS_ML, PHITS_3L

from app.components.model_loader  import load_registry, load_model
from app.components.phits_matcher import find_single, find_multilayer
from app.components.flux_plot     import flux_figure
from app.components.github_write  import (
    github_write_available, create_branch, put_file, open_pull_request,
    new_branch_name, get_file_text,
)

from scripts.inference.single_layer import predict_flux, compute_dose
from scripts.inference.multilayer import (
    predict_chained_cnn_source, predict_tracknet_source, predict_bimodel,
)
from scripts.utilities.check_phits_upload import check_one
from scripts.utilities.source_spectra_editor import (
    insert_new_source, validate_var_name, SourceEditError,
)


def _read_registry_fresh() -> dict:
    """Bypass model_loader's st.cache_resource — needed right after a local
    write (training, or checking for name collisions) so we see the latest
    on-disk state instead of a cached copy from earlier in this deployment's
    lifetime."""
    with open(REPO / "models" / "registry.json") as f:
        return json.load(f)

st.set_page_config(page_title="Model Explorer", page_icon="🔬", layout="wide")

_COMPARE_COLOR = "#9b59b6"   # 2nd-config trace colour (config, not material, so a neutral hue)


# ── password gate ────────────────────────────────────────────────────────────
def _unlocked() -> bool:
    """Tier 2 gate. EXPLORER_PASSWORD lives in .streamlit/secrets.toml (gitignored)."""
    correct = st.secrets.get("EXPLORER_PASSWORD")
    if not correct:
        st.warning(
            "`EXPLORER_PASSWORD` is not set in `.streamlit/secrets.toml` — running "
            "unlocked (local dev mode). Copy `.streamlit/secrets.toml.example` to set "
            "a real password before deploying."
        )
        return True

    if st.session_state.get("explorer_unlocked"):
        return True

    st.title("🔬 Model Explorer")
    st.caption("Collaborator-only: full model documentation and advanced inference tool.")
    pw = st.text_input("Password", type="password", key="explorer_pw_input")
    if st.button("Unlock"):
        if pw == correct:
            st.session_state["explorer_unlocked"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _unlocked():
    st.stop()

st.title("🔬 Model Explorer")
st.caption("Full model registry, technique docs, and advanced inference across all variants.")

registry = load_registry()

tab_docs, tab_infer, tab_code, tab_upload, tab_source, tab_train = st.tabs([
    "📖 Model documentation", "🧪 Advanced inference", "📄 Source code",
    "📤 Upload PHITS", "🧬 Add source spectrum", "🏋️ Train model",
])

# ══════════════════════════════════ TAB 1 — DOCS ═══════════════════════════════
with tab_docs:
    st.subheader("Techniques")

    with st.expander("What is multilayer (chained) inference?"):
        st.markdown(
            "A compound shield (e.g. Concrete 45cm → Steel 33cm) is predicted in two "
            "passes. Layer 1 runs the single-layer model on the TrackNet10 source to get "
            "a transmitted flux; that flux informs layer 2. Several chaining strategies "
            "differ in **what spectrum layer 2 is actually fed** — see the variants "
            "below in the inference tab."
        )
    with st.expander("Why TrackNet10 as the layer-2 source, instead of the layer-1 output?"):
        st.markdown(
            "The normalised layer-1 CNN output is spectrally out-of-distribution for a "
            "model trained on smooth TrackNet10 spectra — feeding it back in as layer 2's "
            "source caused 3–6× amplitude overestimates for concrete-first stacks "
            "(Variant A). Feeding the original TrackNet10 beam keeps layer 2 "
            "in-distribution; the layer-1 amplitude information is preserved separately "
            "via the **a-factor** and multiplied back in at the end (Variant B, primary)."
        )
    with st.expander("What is the a-factor?"):
        st.markdown(
            "`a = sum(layer1_flux[20:150 MeV]) / sum(void_flux[20:150 MeV])` — the "
            "fraction of the unshielded beam that layer 1 transmits, integrated over the "
            "20–150 MeV window (chosen because it's clean signal: below 20 MeV is "
            "thermal-neutron noise, above 150 MeV is near the noise floor). The final "
            "multilayer output is `layer2_flux × a`, restoring the absolute scale that "
            "layer 2 alone can't know since it never saw the shielded beam."
        )

    st.divider()

    section_titles = {
        "primary":      "🟢 Primary models",
        "experimental": "🟡 Experimental models",
        "archived":     "⚪ Archived models",
    }
    for section, heading in section_titles.items():
        models = registry.get(section, {})
        if not models:
            continue
        st.subheader(f"{heading}  ({len(models)})")
        for name, meta in models.items():
            arch_short = (meta.get("architecture") or "")[:70]
            with st.expander(f"`{name}` — {arch_short}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Params", f"{meta.get('params', 0):,}")
                val_mae = meta.get("val_mae")
                c2.metric("Val MAE", f"{val_mae:.3f}" if val_mae is not None else "—")
                c3.metric("Status", meta.get("status", "—"))

                st.markdown(f"**Architecture:** {meta.get('architecture', '—')}")
                st.markdown(f"**Training source:** {meta.get('training_source', '—')}")
                st.markdown(
                    f"**Loss:** {meta.get('loss', '—')}  •  "
                    f"**Created:** {meta.get('created', '—')}  •  "
                    f"**Used for:** {', '.join(meta.get('use_for', [])) or '—'}"
                )
                if meta.get("description"):
                    st.write(meta["description"])

# ═════════════════════════════ TAB 2 — ADVANCED INFERENCE ═════════════════════
with tab_infer:

    def _single_layer_options() -> dict[str, str]:
        """{display label -> pkl path} for every model usable in single-layer mode."""
        opts = {}
        for section in ("primary", "experimental", "archived"):
            for name, meta in registry.get(section, {}).items():
                if "single_layer" in meta.get("use_for", []):
                    opts[f"{name}  [{section}]"] = meta["pkl"]
        return opts

    def _layer2_options() -> dict[str, str]:
        """{display label -> pkl path} for every model usable as multilayer layer-2."""
        opts = {}
        for section in ("primary", "experimental"):
            for name, meta in registry.get(section, {}).items():
                if any(u.startswith("multilayer_layer2") for u in meta.get("use_for", [])):
                    opts[f"{name}  [{section}]"] = meta["pkl"]
        return opts

    SINGLE_OPTS = _single_layer_options()
    LAYER2_OPTS = _layer2_options()
    MAT_NAMES   = list(MATERIALS.keys())

    VARIANT_LABELS = {
        "B": "B — TrackNet10 source (★ primary / recommended)",
        "A": "A — chained CNN source (baseline, known to overshoot)",
        "D": "D — bi-model (source-matched layer-2)",
    }

    def render_config(ns: str):
        """Render config widgets under namespace `ns`; return a config dict."""
        mode = st.radio("Mode", ["Single layer", "Double layer"], key=f"{ns}_mode")
        mat1   = st.selectbox("Material 1", MAT_NAMES, key=f"{ns}_mat1")
        thick1 = st.number_input("Thickness 1 (cm)", min_value=1, max_value=300,
                                  value=45, step=1, key=f"{ns}_thick1")

        if mode == "Single layer":
            model_label = st.selectbox("Model", list(SINGLE_OPTS), key=f"{ns}_model")
            return {"ns": ns, "mode": mode, "mat1": mat1, "thick1": thick1,
                     "model_label": model_label, "model_pkl": SINGLE_OPTS[model_label]}

        # Double layer
        variant_label = st.selectbox("Multilayer variant", list(VARIANT_LABELS.values()),
                                      key=f"{ns}_variant")
        variant = [k for k, v in VARIANT_LABELS.items() if v == variant_label][0]

        mat2   = st.selectbox("Material 2", MAT_NAMES, index=min(1, len(MAT_NAMES) - 1),
                               key=f"{ns}_mat2")
        thick2 = st.number_input("Thickness 2 (cm)", min_value=1, max_value=300,
                                  value=33, step=1, key=f"{ns}_thick2")

        l1_label = st.selectbox("Layer-1 model", list(SINGLE_OPTS), key=f"{ns}_l1model")
        cfg = {"ns": ns, "mode": mode, "variant": variant,
               "mat1": mat1, "thick1": thick1, "mat2": mat2, "thick2": thick2,
               "l1_label": l1_label, "l1_pkl": SINGLE_OPTS[l1_label]}

        if variant == "B":
            l2_label = st.selectbox("Layer-2 model (transfer head)", list(LAYER2_OPTS),
                                     key=f"{ns}_l2model")
            cfg["l2_label"], cfg["l2_pkl"] = l2_label, LAYER2_OPTS[l2_label]
        elif variant == "D":
            l2_label = st.selectbox("Layer-2 model (bi-model, single-layer arch)",
                                     list(SINGLE_OPTS), key=f"{ns}_l2model_bi")
            cfg["l2_label"], cfg["l2_pkl"] = l2_label, SINGLE_OPTS[l2_label]
        # Variant A reuses the layer-1 model for layer 2 — no extra picker.

        return cfg

    def run_config(cfg: dict):
        """Load models on demand (cached) and run inference. Returns (flux, label)."""
        if cfg["mode"] == "Single layer":
            model = load_model(cfg["model_pkl"])
            flux  = predict_flux(model, cfg["mat1"], cfg["thick1"])
            return flux, cfg["model_label"]

        model_v1 = load_model(cfg["l1_pkl"])
        variant  = cfg["variant"]

        if variant == "A":
            flux  = predict_chained_cnn_source(
                model_v1, cfg["mat1"], cfg["thick1"], cfg["mat2"], cfg["thick2"])
            label = f"A: {cfg['l1_label']} (chained)"
        elif variant == "B":
            model_l2 = load_model(cfg["l2_pkl"])
            flux  = predict_tracknet_source(
                model_v1, model_l2, cfg["mat1"], cfg["thick1"], cfg["mat2"], cfg["thick2"])
            label = f"B: {cfg['l1_label']} + {cfg['l2_label']}"
        else:  # D — bimodel
            model_l2 = load_model(cfg["l2_pkl"])
            flux  = predict_bimodel(
                model_v1, model_l2, cfg["mat1"], cfg["thick1"], cfg["mat2"], cfg["thick2"])
            label = f"D: {cfg['l1_label']} + {cfg['l2_label']} (bi-model)"

        return flux, label

    st.subheader("Configuration")
    compare = st.checkbox("Compare two configurations side-by-side")

    if compare:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Configuration A**")
            cfg_a = render_config("A")
        with col_b:
            st.markdown("**Configuration B**")
            cfg_b = render_config("B")
        st.caption(
            "PHITS overlay (if available) and the plot title use Configuration A's "
            "geometry. Give both configs the same material/thickness to isolate the "
            "effect of the model choice."
        )
    else:
        cfg_a = render_config("A")
        cfg_b = None

    run = st.button("▶  Run inference", type="primary")

    if not run:
        # NOTE: deliberately no st.stop() here — st.tabs() runs every tab's body
        # in one linear script pass, so st.stop() here would abort the tabs
        # defined after this one (Source code / Upload PHITS / Add source /
        # Train model) on every page load until this button is clicked once.
        st.info("Configure one or two model setups above and click **▶ Run inference**.")
    else:
        infer_ok = True
        with st.spinner("Computing…"):
            try:
                flux_a, label_a = run_config(cfg_a)
                dose_a = compute_dose(flux_a)
            except Exception as e:
                st.error(f"Configuration A failed: {e}")
                infer_ok = False

            flux_b = label_b = dose_b = None
            if infer_ok and cfg_b is not None:
                try:
                    flux_b, label_b = run_config(cfg_b)
                    dose_b = compute_dose(flux_b)
                except Exception as e:
                    st.error(f"Configuration B failed: {e}")
                    infer_ok = False

            if infer_ok:
                if cfg_a["mode"] == "Single layer":
                    phits_path = find_single(cfg_a["mat1"], int(cfg_a["thick1"]))
                    title = f"{cfg_a['mat1']}  {cfg_a['thick1']} cm"
                else:
                    phits_path = find_multilayer(cfg_a["mat1"], int(cfg_a["thick1"]),
                                                  cfg_a["mat2"], int(cfg_a["thick2"]))
                    title = (f"{cfg_a['mat1']} {cfg_a['thick1']} cm  +  "
                             f"{cfg_a['mat2']} {cfg_a['thick2']} cm")

                phits_flux = dose_phits = None
                if phits_path is not None:
                    phits_flux, _ = load_phits(str(phits_path))
                    if phits_flux is not None:
                        dose_phits = compute_dose(phits_flux)

        if infer_ok:
            col_plot, col_stats = st.columns([3, 1])

            with col_plot:
                fig = flux_figure(
                    energies    = E_ALL,
                    cnn_flux    = flux_a,
                    cnn_label   = label_a,
                    cnn_color   = "#4e79a7",
                    cnn_flux2   = flux_b,
                    cnn_label2  = label_b,
                    cnn_color2  = _COMPARE_COLOR if flux_b is not None else None,
                    phits_flux  = phits_flux,
                    dose_cnn    = dose_a,
                    dose_cnn2   = dose_b,
                    dose_phits  = dose_phits,
                    title       = title,
                )
                st.pyplot(fig, width="stretch")

            with col_stats:
                st.metric(f"Dose — {label_a}", f"{dose_a:.3e} mrem/hr")
                if dose_b is not None:
                    st.metric(f"Dose — {label_b}", f"{dose_b:.3e} mrem/hr")
                if dose_phits is not None:
                    st.metric("Dose — PHITS", f"{dose_phits:.3e} mrem/hr")
                    pct_a = 100.0 * (dose_a - dose_phits) / dose_phits
                    st.metric("Error A", f"{'+' if pct_a >= 0 else ''}{pct_a:.1f}%")
                    if dose_b is not None:
                        pct_b = 100.0 * (dose_b - dose_phits) / dose_phits
                        st.metric("Error B", f"{'+' if pct_b >= 0 else ''}{pct_b:.1f}%")
                else:
                    st.info("No PHITS reference found for this geometry.")

# ══════════════════════════════════ TAB 3 — SOURCE CODE ═══════════════════════
with tab_code:
    st.subheader("Source code")
    st.caption("Read-only — the actual .py files that do inference and training.")

    CODE_FILES = {
        "Single-layer inference — scripts/inference/single_layer.py":
            "scripts/inference/single_layer.py",
        "Multilayer inference, all variants — scripts/inference/multilayer.py":
            "scripts/inference/multilayer.py",
        "Train k100s2 — scripts/training/train_k100s2.py":
            "scripts/training/train_k100s2.py",
        "Train transfer head — scripts/training/train_transfer.py":
            "scripts/training/train_transfer.py",
        "Registry read/write helpers — scripts/training/registry_utils.py":
            "scripts/training/registry_utils.py",
        "Source spectra (fit + registry) — src/shielding_ml/data/source_spectra.py":
            "src/shielding_ml/data/source_spectra.py",
        "Dose computation — src/shielding_ml/metrics/dose.py":
            "src/shielding_ml/metrics/dose.py",
        "PHITS/spectrum loaders — src/shielding_ml/data/loaders.py":
            "src/shielding_ml/data/loaders.py",
    }
    choice = st.selectbox("File", list(CODE_FILES), key="code_file_choice")
    rel_path = CODE_FILES[choice]
    full_path = REPO / rel_path
    if full_path.exists():
        st.code(full_path.read_text(), language="python", line_numbers=True)
    else:
        st.error(f"File not found: {rel_path}")

# ══════════════════════════════════ TAB 4 — UPLOAD PHITS ══════════════════════
with tab_upload:
    st.subheader("Upload a PHITS reference file")
    st.caption(
        "Opens a pull request rather than writing directly — CI re-validates it "
        "(`.github/workflows/phits_check.yml`) and a collaborator reviews before merge."
    )

    if not github_write_available():
        st.info(
            "Write actions need `GITHUB_PAT` set in `.streamlit/secrets.toml` "
            "(see `.streamlit/secrets.toml.example`). Until then, add files by "
            "hand — see [CONTRIBUTING.md](CONTRIBUTING.md)."
        )
    else:
        PHITS_KINDS = {
            "Single layer": (PHITS_HE, "high_energy",
                              "`<Material>_<thickness>cm.out`", "Concrete_45cm.out"),
            "Multilayer":   (PHITS_ML, "multilayer",
                              "`<Mat1>_<t1>cm_<Mat2>_<t2>cm.out`",
                              "Concrete_45cm_Steel_33cm.out"),
            "Three layer":  (PHITS_3L, "3layer",
                              "`<Mat1>_<t1>cm_<Mat2>_<t2>cm_<Mat3>_<t3>cm.out`",
                              "Steel_40cm_Concrete_85cm_BPE_20cm.out"),
        }
        kind = st.radio("Type", list(PHITS_KINDS), key="phits_kind", horizontal=True)
        target_dir, subfolder, convention, example = PHITS_KINDS[kind]

        uploaded = st.file_uploader("PHITS .out file", type=["out"], key="phits_uploader")

        if uploaded is not None:
            st.caption(f"Naming convention: {convention} (e.g. `{example}`)")
            filename = st.text_input("Filename", value=uploaded.name, key="phits_filename")
            content_bytes = uploaded.getvalue()

            if st.button("Validate", key="phits_validate_btn"):
                target_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = target_dir / filename
                tmp_path.write_bytes(content_bytes)
                try:
                    problems = check_one(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

                if problems:
                    st.session_state["phits_valid_for"] = None
                    st.error("Failed validation:")
                    for p in problems:
                        st.write(f"- {p}")
                else:
                    st.session_state["phits_valid_for"] = filename
                    st.session_state["phits_target_rel"] = str(
                        Path("data/raw/phits") / subfolder / filename
                    )
                    st.success("Passes naming + parse checks (250 bins).")

            if st.session_state.get("phits_valid_for") == filename:
                if st.button("📤 Open PR to add this file", type="primary",
                              key="phits_pr_btn"):
                    with st.spinner("Opening pull request…"):
                        try:
                            branch = new_branch_name("add-phits")
                            create_branch(branch)
                            put_file(branch, st.session_state["phits_target_rel"],
                                      content_bytes,
                                      message=f"Add PHITS reference: {filename}")
                            url = open_pull_request(
                                branch,
                                title=f"Add PHITS reference: {filename}",
                                body=(f"Uploaded via Model Explorer.\n\n"
                                      f"Target: `{st.session_state['phits_target_rel']}`\n\n"
                                      f"CI (`phits_check.yml`) will re-validate naming + "
                                      f"parsing on this PR."),
                            )
                            st.success(f"Opened [{url}]({url})")
                            st.session_state["phits_valid_for"] = None
                        except Exception as e:
                            st.error(f"Failed to open PR: {e}")

# ═══════════════════════════════ TAB 5 — ADD SOURCE SPECTRUM ══════════════════
with tab_source:
    st.subheader("Add a new source spectrum")
    st.caption(
        "Fits a piecewise low/high-energy model + noise envelope to an existing "
        "PHITS file and registers it in `SOURCE_FUNCTIONS`. Once the PR is merged "
        "and the app redeploys, train against it with `--source <name>`."
    )

    if not github_write_available():
        st.info(
            "Write actions need `GITHUB_PAT` set in `.streamlit/secrets.toml`. "
            "Until then, see [CONTRIBUTING.md](CONTRIBUTING.md) to add one by hand "
            "(it's a two-line change)."
        )
    else:
        available_phits = sorted(p.name for p in PHITS_HE.glob("*.out"))
        if not available_phits:
            st.warning("No PHITS files found under data/raw/phits/high_energy/ to fit from — "
                       "upload one first in the previous tab.")
        else:
            phits_file = st.selectbox("PHITS file to fit from", available_phits,
                                       key="src_phits_file")
            var_name = st.text_input("New source name (e.g. concrete100)",
                                      key="src_var_name")
            e_cut = st.number_input(
                "E_cut (MeV) — boundary between low- and high-energy fit regimes",
                value=10.0, min_value=1.0, max_value=100.0, key="src_e_cut")
            fit_label = st.text_input(
                "Label for the fitting log message",
                value=phits_file.replace(".out", "").replace("_", " "),
                key="src_fit_label")

            if st.button("Preview", key="src_preview_btn"):
                try:
                    current_text = get_file_text("src/shielding_ml/data/source_spectra.py")
                    existing_names = set(re.findall(
                        r"^(\w+)\s*=\s*_make_noisy_source", current_text, re.M))
                    existing_names.add("tracknet10")
                    validate_var_name(var_name, existing_names)
                    new_text = insert_new_source(current_text, var_name, phits_file,
                                                  float(e_cut), fit_label)
                    st.session_state["src_preview_text"] = new_text
                    st.session_state["src_preview_diff"] = "\n".join(difflib.unified_diff(
                        current_text.splitlines(), new_text.splitlines(),
                        lineterm="", fromfile="before", tofile="after"))
                    st.session_state["src_preview_name"] = var_name
                    st.success("Preview ready — review the diff below, then open the PR.")
                except SourceEditError as e:
                    st.session_state.pop("src_preview_diff", None)
                    st.error(str(e))
                except Exception as e:
                    st.session_state.pop("src_preview_diff", None)
                    st.error(f"Couldn't fetch current source_spectra.py: {e}")

            if st.session_state.get("src_preview_diff"):
                st.code(st.session_state["src_preview_diff"], language="diff")

                if st.button("🧬 Open PR to add this source", type="primary",
                              key="src_pr_btn"):
                    with st.spinner("Opening pull request…"):
                        try:
                            name = st.session_state["src_preview_name"]
                            branch = new_branch_name("add-source")
                            create_branch(branch)
                            put_file(branch, "src/shielding_ml/data/source_spectra.py",
                                      st.session_state["src_preview_text"].encode("utf-8"),
                                      message=f"Add source spectrum: {name}")
                            url = open_pull_request(
                                branch,
                                title=f"Add source spectrum: {name}",
                                body=(f"Fitted from `{phits_file}` (E_cut={e_cut}). "
                                      f"Opened via Model Explorer."),
                            )
                            st.success(f"Opened [{url}]({url})")
                            st.session_state.pop("src_preview_diff", None)
                            st.session_state.pop("src_preview_text", None)
                        except Exception as e:
                            st.error(f"Failed to open PR: {e}")

# ══════════════════════════════════ TAB 6 — TRAIN MODEL ═══════════════════════
with tab_train:
    st.subheader("Train a new model")
    st.caption(
        "Runs synchronously in this session — keep this tab open. On success, the "
        "trained model + updated registry are pushed as a PR for review, never "
        "merged automatically. Training is CPU-heavy; for long runs prefer training "
        "locally (see CONTRIBUTING.md) and treat this panel as the review step."
    )

    if not github_write_available():
        st.info(
            "Write actions need `GITHUB_PAT` set in `.streamlit/secrets.toml`. "
            "Until then, train locally — see [CONTRIBUTING.md](CONTRIBUTING.md)."
        )
    else:
        model_kind = st.radio(
            "Model type",
            ["k100s2 (single-layer)", "Transfer head (multilayer layer-2)"],
            key="train_kind")
        train_name = st.text_input("Model name (must be unique in the registry)",
                                    key="train_name")

        source = arch = None
        if model_kind.startswith("k100s2"):
            from shielding_ml.data.source_spectra import SOURCE_FUNCTIONS
            source = st.selectbox("Source spectrum", list(SOURCE_FUNCTIONS.keys()),
                                   key="train_source")
        else:
            arch = st.selectbox("Head architecture", ["mlp", "linear", "conv"],
                                 key="train_arch")

        epochs = st.number_input("Epochs", min_value=1, max_value=50, value=10,
                                  key="train_epochs")
        if epochs > 25:
            st.warning("More than ~25 epochs can take a long time on shared hosting — "
                       "consider training locally instead for large runs.")
        primary = st.checkbox(
            "Save as primary (replaces the default model the Main Tool uses)",
            value=False, key="train_primary")

        if st.button("🏋️ Start training", type="primary", key="train_start_btn"):
            registry_now = _read_registry_fresh()
            existing_section = next(
                (s for s in ("primary", "experimental", "archived")
                 if train_name in registry_now.get(s, {})), None)

            if not train_name:
                st.error("Model name is required.")
            elif existing_section is not None:
                st.error(f"'{train_name}' already exists in registry section "
                         f"'{existing_section}'. Pick a different name.")
            else:
                argv = ["prog", "--name", train_name, "--epochs", str(int(epochs))]
                if primary:
                    argv.append("--primary")
                if model_kind.startswith("k100s2"):
                    argv += ["--source", source]
                    module_name = "scripts.training.train_k100s2"
                else:
                    argv += ["--arch", arch]
                    module_name = "scripts.training.train_transfer"

                import contextlib
                import importlib
                import io

                mod = importlib.import_module(module_name)
                log_buf = io.StringIO()
                old_argv = sys.argv
                sys.argv = argv
                rc = None
                try:
                    with st.spinner(f"Training '{train_name}' — this can take "
                                     f"several minutes…"):
                        with contextlib.redirect_stdout(log_buf):
                            rc = mod.main()
                except (Exception, SystemExit) as e:
                    st.error(f"Training crashed: {e}")
                finally:
                    sys.argv = old_argv

                st.text_area("Training log", log_buf.getvalue(), height=300,
                              key="train_log_out")

                if rc == 0:
                    st.success(f"Trained '{train_name}' successfully. Pushing to a PR…")
                    try:
                        new_registry = _read_registry_fresh()
                        section = "primary" if primary else "experimental"
                        entry = new_registry[section][train_name]
                        model_dir = (REPO / entry["pkl"]).parent

                        with st.spinner("Pushing model files to GitHub…"):
                            branch = new_branch_name(f"train-{train_name}")
                            create_branch(branch)

                            pkl_path = REPO / entry["pkl"]
                            put_file(branch, entry["pkl"], pkl_path.read_bytes(),
                                      message=f"Add trained model: {train_name}")

                            desc_path = model_dir / "description.txt"
                            if desc_path.exists():
                                rel = str(desc_path.relative_to(REPO))
                                put_file(branch, rel, desc_path.read_bytes(),
                                          message=f"Add description for {train_name}")

                            loss_curve = model_dir / "loss_curve.png"
                            if loss_curve.exists():
                                rel = str(loss_curve.relative_to(REPO))
                                put_file(branch, rel, loss_curve.read_bytes(),
                                          message=f"Add loss curve for {train_name}")

                            registry_bytes = (REPO / "models" / "registry.json").read_bytes()
                            put_file(branch, "models/registry.json", registry_bytes,
                                      message=f"Register {train_name} in models/registry.json")

                            url = open_pull_request(
                                branch,
                                title=f"Add trained model: {train_name}",
                                body=(f"Trained via Model Explorer.\n\n"
                                      f"- Type: {model_kind}\n"
                                      f"- {'Source: ' + source if source else 'Architecture: ' + arch}\n"
                                      f"- Epochs: {epochs}\n"
                                      f"- Section: {section}\n\n"
                                      f"Review before merging — this becomes live for every "
                                      f"user once merged."),
                            )
                        st.success(f"Opened [{url}]({url})")
                    except Exception as e:
                        st.error(f"Trained locally but failed to push to GitHub: {e}. "
                                 f"The model files are still on this container's local "
                                 f"disk (ephemeral — lost on redeploy) at {model_dir}.")
                elif rc is not None:
                    st.error(f"Training script exited with an error (code {rc}) — "
                             f"see the log above.")
