"""
Validate models/registry.json — schema + referenced files exist.

Run manually:
    ../venv/bin/python scripts/utilities/validate_registry.py

Used by CI (.github/workflows/ci.yml) on every push/PR so a bad registry
entry (typo'd path, missing required field) fails the build instead of
silently breaking one of the Streamlit apps at runtime.

Exit code 0 = valid, 1 = errors found (printed to stderr).
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

REQUIRED_FIELDS = ("pkl", "architecture", "params", "created", "status", "description")
VALID_STATUS    = {"active", "experimental", "archived"}


def validate(registry: dict) -> list[str]:
    errors: list[str] = []
    seen_names: dict[str, str] = {}   # name -> section, to catch cross-section duplicates

    for key in registry:
        if key == "_comment":
            continue
        if key not in SECTIONS:
            errors.append(f"Unknown top-level section '{key}' (expected one of {SECTIONS})")

    for section in SECTIONS:
        models = registry.get(section, {})
        if not isinstance(models, dict):
            errors.append(f"Section '{section}' must be an object")
            continue

        for name, meta in models.items():
            loc = f"{section}.{name}"

            if name in seen_names:
                errors.append(f"{loc}: duplicate model name — already registered "
                               f"under '{seen_names[name]}'")
            else:
                seen_names[name] = section

            if not isinstance(meta, dict):
                errors.append(f"{loc}: entry must be an object")
                continue

            for field in REQUIRED_FIELDS:
                if field not in meta:
                    errors.append(f"{loc}: missing required field '{field}'")

            pkl = meta.get("pkl")
            if pkl:
                if not (REPO_ROOT / pkl).exists():
                    errors.append(f"{loc}: pkl path does not exist: {pkl}")

            params = meta.get("params")
            if params is not None and not isinstance(params, int):
                errors.append(f"{loc}: 'params' must be an integer, got {type(params).__name__}")

            status = meta.get("status")
            if status is not None and status not in VALID_STATUS:
                errors.append(f"{loc}: 'status' must be one of {VALID_STATUS}, got {status!r}")

            val_mae = meta.get("val_mae")
            if val_mae is not None and not isinstance(val_mae, (int, float)):
                errors.append(f"{loc}: 'val_mae' must be numeric or null, got "
                               f"{type(val_mae).__name__}")

            use_for = meta.get("use_for")
            if use_for is not None and not isinstance(use_for, list):
                errors.append(f"{loc}: 'use_for' must be a list")

    return errors


def main() -> int:
    if not REGISTRY_PATH.exists():
        print(f"ERROR: {REGISTRY_PATH} does not exist", file=sys.stderr)
        return 1

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    errors = validate(registry)

    if errors:
        print(f"models/registry.json — {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return 1

    n_models = sum(len(registry.get(s, {})) for s in SECTIONS)
    print(f"models/registry.json OK — {n_models} models across "
          f"{sum(1 for s in SECTIONS if registry.get(s))} sections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
