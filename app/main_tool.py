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

Layout: "scientific dashboard" — top header (branding + page nav + status),
a geometry card beside the flux plot, three result cards (CNN / PHITS / error)
as the focal point, and secondary detail collapsed into expanders at the
bottom. Deliberately not the sidebar-heavy layout Streamlit defaults to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shielding_ml.data.constants import MATERIALS, E_ALL
from shielding_ml.data.loaders import load_phits

from app.components.model_loader  import primary_models
from app.components.phits_matcher import (
    find_single, find_multilayer, available_single_cases, available_multilayer_cases,
)
from app.components.flux_plot     import flux_figure, mat_color

from scripts.inference.single_layer import predict_flux, compute_dose
from scripts.inference.multilayer   import predict_tracknet_source

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Shielding CNN",
    page_icon   = "☢️",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── formatting helper ────────────────────────────────────────────────────────
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def format_sci(value: float, sig: int = 3) -> str:
    """'2.26 × 10⁻¹' style scientific notation, not Python's '2.26e-01'."""
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exp)
    if round(mantissa, sig - 1) >= 10:
        mantissa /= 10
        exp += 1
    return f"{mantissa:.{sig - 1}f} × 10{str(exp).translate(_SUPERSCRIPT)}"


# ── design system: spacing scale + hide the default page sidebar nav ──────────
st.markdown("""
<style>
    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarCollapseButton"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 4px;
    }

    .app-header {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 14px; margin-bottom: 8px;
        border-bottom: 1px solid rgba(140,21,21,0.15);
    }
    .app-header .brand { display: flex; align-items: center; gap: 10px; }
    .app-header .brand img { height: 22px; width: auto; }
    .app-header .brand .name { font-weight: 700; font-size: 16px; color: #1E293B; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 13px; color: #2f6f4f; background: #e9f7ef;
        padding: 4px 12px; border-radius: 999px; font-weight: 600;
    }
    .status-pill .dot {
        width: 8px; height: 8px; border-radius: 50%; background: #2f9e5c;
        display: inline-block;
    }

    .page-title { font-size: 34px; font-weight: 800; margin-bottom: 2px; color: #1E293B; }
    .page-subtitle { font-size: 15px; color: #64748B; margin-bottom: 28px; }

    .card-label {
        font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
        color: #64748B; text-transform: uppercase; margin-bottom: 10px;
    }
    .case-label { font-size: 13px; color: #64748B; margin-bottom: 2px; }

    .metric-card {
        border-radius: 12px; padding: 20px 22px; height: 100%;
    }
    .metric-card .metric-label {
        font-size: 12px; font-weight: 700; letter-spacing: 0.05em;
        text-transform: uppercase; margin-bottom: 8px;
    }
    .metric-card .metric-value { font-size: 26px; font-weight: 800; line-height: 1.15; }
    .metric-card .metric-unit { font-size: 13px; font-weight: 500; margin-left: 4px; }
    .metric-card .metric-sub { font-size: 12.5px; margin-top: 8px; font-weight: 600; }

    .metric-primary  { background: #f0f6ff; border: 1px solid #cfe1fb; }
    .metric-primary  .metric-label, .metric-primary .metric-unit { color: #2563EB; }
    .metric-primary  .metric-value { color: #1e3a8a; }

    .metric-secondary { background: #f8fafc; border: 1px solid #e2e8f0; }
    .metric-secondary .metric-label, .metric-secondary .metric-unit { color: #64748B; }
    .metric-secondary .metric-value { color: #334155; }

    .metric-ok    { background: #eefbf2; border: 1px solid #bfe8cd; }
    .metric-ok    .metric-label { color: #1a7f4b; }
    .metric-ok    .metric-value { color: #1a7f4b; }
    .metric-ok    .metric-sub   { color: #1a7f4b; }

    .metric-warn  { background: #fffaeb; border: 1px solid #fbe4a6; }
    .metric-warn  .metric-label { color: #a15c00; }
    .metric-warn  .metric-value { color: #a15c00; }
    .metric-warn  .metric-sub   { color: #a15c00; }

    .metric-bad   { background: #fef2f2; border: 1px solid #f6c2c2; }
    .metric-bad   .metric-label { color: #b91c1c; }
    .metric-bad   .metric-value { color: #b91c1c; }
    .metric-bad   .metric-sub   { color: #b91c1c; }
</style>
""", unsafe_allow_html=True)

# ── load models (cached — runs once per session) ──────────────────────────────
@st.cache_resource(show_spinner="Loading CNN models…")
def get_models():
    return primary_models()


model_v1, model_transfer = get_models()

# ── top header: branding · page nav · status ───────────────────────────────────
logo_path = REPO / "app" / "assets" / "slac_logo.png"
col_brand, col_nav, col_status = st.columns([3, 4, 2])
with col_brand:
    if logo_path.exists():
        import base64
        _logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        st.markdown(
            f'<div class="app-header" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">'
            f'<div class="brand">'
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="height:22px;width:auto;max-width:none;display:block;"/>'
            f'<span class="name">☢ Shielding CNN</span></div></div>',
            unsafe_allow_html=True,
        )
with col_nav:
    n1, n2 = st.columns(2)
    with n1:
        st.page_link("main_tool.py", label="**Predictor**", icon=":material/my_location:")
    with n2:
        st.page_link("pages/2_Model_Explorer.py", label="Model Explorer",
                      icon=":material/science:")
with col_status:
    st.markdown(
        '<div style="text-align:right; padding-top:6px;">'
        '<span class="status-pill"><span class="dot"></span>Model ready</span></div>',
        unsafe_allow_html=True,
    )
st.markdown('<hr style="margin-top:8px; margin-bottom:24px; opacity:0.15;">',
            unsafe_allow_html=True)

# ── page title ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Neutron Flux Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Configure shielding geometry and compare CNN vs PHITS — '
            'updates live, no button to click.</div>', unsafe_allow_html=True)

# ── geometry card + flux spectrum ──────────────────────────────────────────────
col_geo, col_plot = st.columns([1, 2.3], gap="medium")

with col_geo:
    with st.container(border=True):
        st.markdown('<div class="card-label">Geometry</div>', unsafe_allow_html=True)

        mode = st.segmented_control(
            "Mode", options=["Single layer", "Double layer"],
            default="Single layer", key="mode", label_visibility="collapsed",
        )
        if mode is None:
            mode = "Single layer"

        st.markdown("**Layer 1**")
        mat1 = st.selectbox("Material", list(MATERIALS.keys()), key="mat1")
        thick1 = st.slider("Thickness (cm)", min_value=1, max_value=300,
                            value=45, key="thick1")

        if mode == "Double layer":
            st.markdown("**Layer 2**")
            mat2 = st.selectbox("Material", list(MATERIALS.keys()), index=1, key="mat2")
            thick2 = st.slider("Thickness (cm)", min_value=1, max_value=300,
                                value=33, key="thick2")

with col_plot:
    with st.container(border=True):
        if mode == "Single layer":
            case_label = f"{mat1} · {thick1} cm"
        else:
            case_label = f"{mat1} {thick1} cm  +  {mat2} {thick2} cm"
        st.markdown(f'<div class="case-label">{case_label}</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Flux spectrum</div>', unsafe_allow_html=True)

        # ── run inference (reactive — reruns on every widget change) ──────────
        with st.spinner("Computing…"):
            if mode == "Single layer":
                cnn_flux  = predict_flux(model_v1, mat1, thick1)
                dose_cnn  = compute_dose(cnn_flux)
                color     = mat_color(mat1)
                cnn_label = "CNN model (k100s2_v1)"
                phits_path = find_single(mat1, int(thick1))
            else:
                cnn_flux  = predict_tracknet_source(
                    model_v1, model_transfer, mat1, thick1, mat2, thick2)
                dose_cnn  = compute_dose(cnn_flux)
                color     = mat_color(mat1)
                cnn_label = "CNN model (k100s2_v1 + transfer_mlp)"
                phits_path = find_multilayer(mat1, int(thick1), mat2, int(thick2))

            phits_flux = dose_phits = None
            if phits_path is not None:
                phits_flux, _ = load_phits(str(phits_path))
                if phits_flux is not None:
                    dose_phits = compute_dose(phits_flux)

        fig = flux_figure(
            energies    = E_ALL,
            cnn_flux    = cnn_flux,
            cnn_label   = cnn_label,
            cnn_color   = color,
            phits_flux  = phits_flux,
            dose_cnn    = dose_cnn,
            dose_phits  = dose_phits,
            show_dose_annotation = False,   # the metric cards below cover this
        )
        st.pyplot(fig, width="stretch")

# ── result cards — the focal point ─────────────────────────────────────────────
st.write("")
if dose_phits is not None:
    pct = 100.0 * (dose_cnn - dose_phits) / dose_phits
    sign = "+" if pct >= 0 else ""
    if abs(pct) <= 20:
        err_class, err_status = "metric-ok", "✓ Within ±20% target"
    elif abs(pct) <= 30:
        err_class, err_status = "metric-warn", "⚠ Within ±30%"
    else:
        err_class, err_status = "metric-bad", "✗ Outside ±30%"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card metric-primary">'
            f'<div class="metric-label">CNN Prediction</div>'
            f'<div class="metric-value">{format_sci(dose_cnn)}<span class="metric-unit">mrem/hr</span></div>'
            f'<div class="metric-sub">Primary result</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card metric-secondary">'
            f'<div class="metric-label">PHITS Reference</div>'
            f'<div class="metric-value">{format_sci(dose_phits)}<span class="metric-unit">mrem/hr</span></div>'
            f'<div class="metric-sub">Monte Carlo ground truth</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card {err_class}">'
            f'<div class="metric-label">Dose Error</div>'
            f'<div class="metric-value">{sign}{pct:.1f}%</div>'
            f'<div class="metric-sub">{err_status}</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f'<div class="metric-card metric-primary" style="max-width:340px;">'
        f'<div class="metric-label">CNN Prediction</div>'
        f'<div class="metric-value">{format_sci(dose_cnn)}<span class="metric-unit">mrem/hr</span></div>'
        f'<div class="metric-sub">No PHITS reference for this geometry</div></div>',
        unsafe_allow_html=True,
    )

# ── secondary detail, collapsed by default ─────────────────────────────────────
st.write("")
with st.expander("Model details"):
    st.markdown(
        "**Single layer:** `k100s2_v1`  \n"
        "**Double layer:** `k100s2_v1` (layer-1 a-factor) → `transfer_mlp` "
        "with TrackNet10 as layer-2 source"
    )
    arch_dir = REPO / "images_for_paper" / "architecture"
    if mode == "Single layer":
        st.image(str(arch_dir / "k100s2_v1.png"), caption="k100s2_v1 — single-layer model",
                  width="stretch")
    else:
        st.image(str(arch_dir / "k100s2_v1.png"), caption="k100s2_v1 — layer-1 (a-factor)",
                  width="stretch")
        st.image(str(arch_dir / "transfer_mlp.png"), caption="transfer_mlp — layer-2",
                  width="stretch")

    st.divider()
    st.markdown("**Available PHITS cases**")
    if mode == "Single layer":
        cases = [f"{m} {t} cm" for m, t in available_single_cases()]
    else:
        cases = [f"{m1} {t1} + {m2} {t2}" for m1, t1, m2, t2 in available_multilayer_cases()]
    st.caption(", ".join(cases) if cases else "None found")

with st.expander("Add new PHITS data"):
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
        "The app will pick up the file automatically on next load. Or use the "
        "**Model Explorer → Upload PHITS** tab to do this without git."
    )
