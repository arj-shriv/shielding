"""
Load models from disk and the registry.

Uses st.cache_resource so each model is loaded once per app session.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

from shielding_ml.pipelines.paths import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


@st.cache_resource(show_spinner="Loading registry…")
def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading model…")
def load_model(rel_path: str):
    """Load a Keras model .pkl by path relative to REPO_ROOT."""
    full = REPO_ROOT / rel_path
    with open(full, "rb") as f:
        return pickle.load(f)


def primary_models():
    """Return (model_v1, model_transfer) — the two primary models."""
    reg = load_registry()
    v1       = load_model(reg["primary"]["k100s2_v1"]["pkl"])
    transfer = load_model(reg["primary"]["transfer_mlp"]["pkl"])
    return v1, transfer


def all_single_layer_models() -> dict[str, object]:
    """
    Return {display_name: model} for every model usable in single-layer mode.
    Includes primary + experimental + archived entries with use_for == 'single_layer'.
    """
    reg    = load_registry()
    result = {}

    for section in ("primary", "experimental", "archived"):
        for name, meta in reg.get(section, {}).items():
            if "single_layer" in meta.get("use_for", []):
                label = f"{name}  [{section}]"
                result[label] = load_model(meta["pkl"])

    return result


def all_layer2_models() -> dict[str, object]:
    """Return {display_name: model} for every model usable as multilayer layer-2."""
    reg    = load_registry()
    result = {}

    for section in ("primary", "experimental"):
        for name, meta in reg.get(section, {}).items():
            uses = meta.get("use_for", [])
            if any(u.startswith("multilayer_layer2") for u in uses):
                label = f"{name}  [{section}]"
                result[label] = load_model(meta["pkl"])

    return result
