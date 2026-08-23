"""
Shielding CNN — Main Inference Tool
====================================
Streamlit app for single-layer and double-layer neutron shielding inference.

Run from the repo root:
    streamlit run app/main_tool.py

Default models
--------------
    Single layer  : k100s2_v1
    Double layer  : k100s2_v1 (layer-1 a-factor) + transfer_mlp (TrackNet10 source)

If a PHITS reference file exists for the selected geometry, it is overlaid
automatically and the dose error is displayed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.data.constants import MATERIALS, E_ALL
from shielding_ml.data.loaders import load_phits

from app.components.model_loader  import primary_models
from app.components.phits_matcher import find_single, find_multilayer
from app.components.flux_plot     import flux_figure, mat_color

from scripts.inference.single_layer import predict_flux, compute_dose
from scripts.inference.multilayer   import predict_tracknet_source

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Shielding CNN",
    page_icon   = "☢️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── load models (cached — runs once per session) ──────────────────────────────
@st.cache_resource(show_spinner="Loading CNN models…")
def get_models():
    return primary_models()


model_v1, model_transfer = get_models()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("☢️ Shielding CNN")
    st.caption("Ca-48 @ 150 MeV/n  •  k100s2 architecture")
    st.divider()

    mode = st.radio(
        "Inference mode",
        options=["Single layer", "Double layer"],
        index=0,
    )

    st.divider()
    st.subheader("Layer 1")
    mat1   = st.selectbox("Material",      list(MATERIALS.keys()), key="mat1")
    thick1 = st.number_input("Thickness (cm)", min_value=1, max_value=300,
                              value=45, step=1, key="thick1")

    if mode == "Double layer":
        st.subheader("Layer 2")
        mat2   = st.selectbox("Material",      list(MATERIALS.keys()),
                               index=1, key="mat2")
        thick2 = st.number_input("Thickness (cm)", min_value=1, max_value=300,
                                  value=33, step=1, key="thick2")

    st.divider()
    run = st.button("▶  Run inference", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️ Models used"):
        st.markdown(
            "**Single layer:** `k100s2_v1`  \n"
            "**Double layer:** `k100s2_v1` (layer-1 a-factor)  \n"
            "→ `transfer_mlp` with TrackNet10 as layer-2 source"
        )

    with st.expander("📂 Add new PHITS data"):
        st.markdown(
            "To add a new reference case so it appears as an overlay here:\n\n"
            "1. Clone the repo:  \n"
            "   `git clone git@github.com:arj-shriv/shielding.git`\n\n"
            "2. Drop your PHITS `.out` file into the correct folder:\n"
            "   - Single layer: `data/raw/phits/high_energy/`\n"
            "   - Double layer: `data/raw/phits/multilayer/`\n\n"
            "3. Name it following the convention:\n"
            "   - `Concrete_45cm.out` or `Concrete_45cm_1.out`\n"
            "   - `Concrete_45cm_Steel_33cm.out`\n\n"
            "4. Commit and push:\n"
            "   ```\n"
            "   git add data/raw/phits/...\n"
            "   git commit -m 'Add PHITS: <case name>'\n"
            "   git push origin main\n"
            "   ```\n\n"
            "The app will pick up the file automatically on next load."
        )

# ── main panel ────────────────────────────────────────────────────────────────
st.header("Neutron Flux Spectrum Prediction")

if not run:
    st.info("Configure a geometry in the sidebar and click **▶ Run inference**.")
    st.stop()

# ── run inference ─────────────────────────────────────────────────────────────
with st.spinner("Computing…"):

    if mode == "Single layer":
        cnn_flux   = predict_flux(model_v1, mat1, thick1)
        dose_cnn   = compute_dose(cnn_flux)
        title      = f"{mat1}  {thick1} cm"
        color      = mat_color(mat1)
        cnn_label  = "CNN model (k100s2_v1)"

        phits_path = find_single(mat1, int(thick1))

    else:  # Double layer
        cnn_flux   = predict_tracknet_source(
            model_v1, model_transfer,
            mat1, thick1, mat2, thick2,
        )
        dose_cnn   = compute_dose(cnn_flux)
        title      = f"{mat1} {thick1} cm  +  {mat2} {thick2} cm"
        color      = mat_color(mat1)
        cnn_label  = "CNN model (k100s2_v1 + transfer_mlp)"

        phits_path = find_multilayer(mat1, int(thick1), mat2, int(thick2))

    # Load PHITS reference if available
    phits_flux  = None
    dose_phits  = None
    if phits_path is not None:
        phits_flux, _ = load_phits(str(phits_path))
        if phits_flux is not None:
            dose_phits = compute_dose(phits_flux)

# ── results ───────────────────────────────────────────────────────────────────
col_plot, col_stats = st.columns([3, 1])

with col_plot:
    fig = flux_figure(
        energies   = E_ALL,
        cnn_flux   = cnn_flux,
        cnn_label  = cnn_label,
        cnn_color  = color,
        phits_flux = phits_flux,
        dose_cnn   = dose_cnn,
        dose_phits = dose_phits,
        title      = title,
    )
    st.pyplot(fig, use_container_width=True)

with col_stats:
    st.metric("CNN dose rate", f"{dose_cnn:.3e} mrem/hr")

    if dose_phits is not None:
        pct = 100.0 * (dose_cnn - dose_phits) / dose_phits
        sign = "+" if pct >= 0 else ""
        st.metric("PHITS dose rate",    f"{dose_phits:.3e} mrem/hr")
        st.metric("Dose error",         f"{sign}{pct:.1f}%",
                  delta=f"{sign}{pct:.1f}%",
                  delta_color="inverse")
        if abs(pct) <= 20:
            st.success("Within ±20% target")
        elif abs(pct) <= 30:
            st.warning("Within ±30%")
        else:
            st.error("Outside ±30%")
    else:
        st.info(
            "No PHITS reference found for this geometry.  \n"
            "See the sidebar for instructions on adding one."
        )

    st.divider()
    st.caption("**Geometry**")
    st.write(f"Material 1: **{mat1}** ({MATERIALS[mat1]} g/cm³)")
    st.write(f"Thickness 1: **{thick1} cm**")
    if mode == "Double layer":
        st.write(f"Material 2: **{mat2}** ({MATERIALS[mat2]} g/cm³)")
        st.write(f"Thickness 2: **{thick2} cm**")

    st.divider()
    st.caption("**Available PHITS cases**")
    from app.components.phits_matcher import (
        available_single_cases, available_multilayer_cases,
    )
    if mode == "Single layer":
        cases = [f"{m} {t} cm" for m, t in available_single_cases()]
    else:
        cases = [f"{m1} {t1}+{m2} {t2}" for m1, t1, m2, t2 in available_multilayer_cases()]

    if cases:
        for c in cases:
            st.caption(f"• {c}")
    else:
        st.caption("None found")
