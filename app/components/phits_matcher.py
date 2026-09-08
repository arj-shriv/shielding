"""
Map user-selected (material, thickness) → PHITS reference file, if one exists.

Scans data/raw/phits/high_energy/ and multilayer/ at import time and builds
lookup dicts. Files are re-scanned if the app is restarted (no persistent cache).

Single-layer key : (material, thickness_cm)  e.g. ('Concrete', 45)
Multilayer key   : (mat1, thick1, mat2, thick2)  e.g. ('Concrete',45,'Steel',33)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shielding_ml.pipelines.paths import PHITS_HE, PHITS_ML


@st.cache_resource(show_spinner=False)
def single_layer_index() -> dict[tuple[str, int], Path]:
    """
    Return {(material, thickness_cm): Path} for available single-layer PHITS files.

    Picks the '_1' variant when multiple files exist for the same case
    (e.g. Concrete_45cm_1.out preferred over Concrete_45cm.out).
    """
    idx: dict[tuple[str, int], Path] = {}
    prefer: dict[tuple[str, int], bool] = {}  # True if already holds a _1 variant

    for f in sorted(PHITS_HE.iterdir()):
        if f.suffix not in {".out", ".txt", ""} or f.name.startswith("."):
            continue
        m = re.match(r"^([A-Za-z]+)_(\d+)cm", f.stem)
        if not m:
            continue
        mat, thick = m.group(1), int(m.group(2))
        key    = (mat, thick)
        is_v1  = bool(re.search(r"_1$", f.stem))
        is_dup = bool(re.search(r"_2$", f.stem))

        if is_dup:
            continue
        if key not in idx or (is_v1 and not prefer.get(key)):
            idx[key]    = f
            prefer[key] = is_v1

    return idx


@st.cache_resource(show_spinner=False)
def multilayer_index() -> dict[tuple[str, int, str, int], Path]:
    """
    Return {(mat1, thick1, mat2, thick2): Path} for multilayer PHITS files.

    Expects names like Concrete_45cm_Steel_33cm.out.
    """
    idx: dict[tuple[str, int, str, int], Path] = {}

    for f in sorted(PHITS_ML.iterdir()):
        if f.suffix != ".out" or f.name.startswith("."):
            continue
        pairs = re.findall(r"([A-Za-z]+)_(\d+)cm", f.stem)
        if len(pairs) == 2:
            (m1, t1), (m2, t2) = pairs
            idx[(m1, int(t1), m2, int(t2))] = f

    return idx


def find_single(material: str, thickness_cm: int) -> Path | None:
    return single_layer_index().get((material, thickness_cm))


def find_multilayer(mat1: str, thick1: int, mat2: str, thick2: int) -> Path | None:
    return multilayer_index().get((mat1, thick1, mat2, thick2))


def available_single_cases() -> list[tuple[str, int]]:
    return sorted(single_layer_index().keys())


def available_multilayer_cases() -> list[tuple[str, int, str, int]]:
    return sorted(multilayer_index().keys())
