"""
Shared helpers for training scripts: registry read/write, the "does this
model already exist" check, and description.txt generation.

Both train_k100s2.py and train_transfer.py import this so models/registry.json
stays the single source of truth that both Streamlit apps read from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shielding_ml.pipelines.paths import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"
SECTIONS = ("primary", "experimental", "archived")


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")


def find_model(name: str, registry: dict | None = None) -> str | None:
    """Return the section a model name is already registered under, or None."""
    registry = registry if registry is not None else load_registry()
    for section in SECTIONS:
        if name in registry.get(section, {}):
            return section
    return None


def upsert_entry(name: str, section: str, entry: dict) -> None:
    """Write/replace one model's registry entry and persist to disk immediately."""
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {SECTIONS}, got {section!r}")
    registry = load_registry()
    registry.setdefault(section, {})[name] = entry
    save_registry(registry)


def write_description(model_dir: Path, description: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "description.txt").write_text(description.strip() + "\n")


def base_model_pkl(registry: dict | None = None) -> Path:
    """Path to the primary k100s2_v1 model.pkl, resolved from the registry."""
    registry = registry if registry is not None else load_registry()
    return REPO_ROOT / registry["primary"]["k100s2_v1"]["pkl"]


def require_available(name: str, overwrite: bool) -> str | None:
    """
    Exit-code-1 guard used at the top of every training script.

    Returns the existing section if the name is already registered and
    --overwrite was not passed (caller should print + sys.exit(1) on non-None).
    Returns None if training may proceed.
    """
    existing = find_model(name)
    if existing is not None and not overwrite:
        return existing
    return None
