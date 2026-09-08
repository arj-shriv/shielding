"""
Shared matplotlib flux-spectrum plot for the Streamlit apps.

Returns a matplotlib Figure that the caller passes to st.pyplot().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Material colour palette (matches images_for_paper scripts)
MAT_COLOR = {
    "Concrete": "#4e79a7",
    "Steel":    "#e15759",
    "BPE":      "#59a14f",
}
_FALLBACK = "#666666"


def flux_figure(
    energies:      np.ndarray,
    cnn_flux:      np.ndarray,
    cnn_label:     str,
    cnn_color:     str,
    phits_flux:    np.ndarray | None = None,
    dose_cnn:      float | None      = None,
    dose_phits:    float | None      = None,
    title:         str               = "",
    figsize:       tuple             = (8, 5),
    cnn_flux2:     np.ndarray | None = None,
    cnn_label2:    str | None        = None,
    cnn_color2:    str | None        = None,
    dose_cnn2:     float | None      = None,
    show_dose_annotation: bool       = True,
) -> plt.Figure:
    """
    Build a semilogy flux spectrum figure.

    Parameters
    ----------
    energies   : 1-D energy array (MeV)
    cnn_flux   : CNN-predicted flux (n/cm²/source)
    cnn_label  : legend label for CNN line
    cnn_color  : hex colour for CNN line
    phits_flux : PHITS reference flux (optional overlay)
    dose_cnn   : CNN dose rate (mrem/hr) — shown in subtitle if provided
    dose_phits : PHITS dose rate (mrem/hr) — shown with error % if provided
    title      : main plot title
    figsize    : figure size in inches
    cnn_flux2/cnn_label2/cnn_color2/dose_cnn2 :
        optional second CNN trace, for side-by-side config comparison
        (Model Explorer). Omit for the single-trace Main Tool plot.
    show_dose_annotation :
        print the dose/error text below the plot. Main Tool sets this False
        since the same numbers live in dedicated metric cards next to the
        plot — repeating them inside the figure is just noise there.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    # PHITS reference (drawn first so CNN overlays on top)
    if phits_flux is not None:
        ax.semilogy(energies, phits_flux, color="black", lw=2.0,
                    label="PHITS", zorder=5)

    # CNN prediction(s)
    ax.semilogy(energies, cnn_flux, color=cnn_color, lw=2.0,
                ls="--", label=cnn_label, zorder=6)
    if cnn_flux2 is not None:
        ax.semilogy(energies, cnn_flux2, color=cnn_color2, lw=2.0,
                    ls=":", label=cnn_label2, zorder=6)

    ax.set_xlim(0, 250)
    ax.set_xlabel("Neutron energy (MeV)", fontsize=11)
    ax.set_ylabel("Flux  [n / cm² / source n]", fontsize=11)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.6)
    ax.legend(fontsize=10)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Dose annotation below x-axis
    if show_dose_annotation and dose_cnn is not None:
        parts = [f"{cnn_label}: {dose_cnn:.3e} mrem/hr"]
        if dose_cnn2 is not None:
            parts.append(f"{cnn_label2}: {dose_cnn2:.3e} mrem/hr")
        if dose_phits is not None:
            pct = 100.0 * (dose_cnn - dose_phits) / dose_phits
            sign = "+" if pct >= 0 else ""
            parts.append(f"PHITS: {dose_phits:.3e} mrem/hr")
            parts.append(f"Error: {sign}{pct:.1f}%")
        subtitle = "Dose rate — " + "  |  ".join(parts)

        fig.text(
            0.5, -0.02, subtitle,
            ha="center", fontsize=9, color="#444444",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    return fig


def mat_color(material: str) -> str:
    return MAT_COLOR.get(material, _FALLBACK)
