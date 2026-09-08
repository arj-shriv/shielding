"""
Tests for the Phase 4 training/registry tooling:
  - scripts/utilities/validate_registry.py
  - scripts/training/registry_utils.py
  - scripts/utilities/check_phits_upload.py

Run from the repo root: pytest tests/
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
REGISTRY_PATH = REPO / "models" / "registry.json"


@pytest.fixture
def real_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


# ── validate_registry ────────────────────────────────────────────────────────

def test_real_registry_is_valid(real_registry):
    from scripts.utilities.validate_registry import validate
    assert validate(real_registry) == []


def test_validate_catches_missing_required_field(real_registry):
    from scripts.utilities.validate_registry import validate
    reg = copy.deepcopy(real_registry)
    del reg["primary"]["k100s2_v1"]["architecture"]
    errors = validate(reg)
    assert any("missing required field 'architecture'" in e for e in errors)


def test_validate_catches_duplicate_name_across_sections(real_registry):
    from scripts.utilities.validate_registry import validate
    reg = copy.deepcopy(real_registry)
    reg["experimental"]["k100s2_v1"] = reg["primary"]["k100s2_v1"]
    errors = validate(reg)
    assert any("duplicate model name" in e for e in errors)


def test_validate_catches_missing_pkl(real_registry):
    from scripts.utilities.validate_registry import validate
    reg = copy.deepcopy(real_registry)
    reg["primary"]["k100s2_v1"]["pkl"] = "models/does/not/exist.pkl"
    errors = validate(reg)
    assert any("pkl path does not exist" in e for e in errors)


def test_validate_catches_bad_status(real_registry):
    from scripts.utilities.validate_registry import validate
    reg = copy.deepcopy(real_registry)
    reg["primary"]["k100s2_v1"]["status"] = "not_a_real_status"
    errors = validate(reg)
    assert any("'status' must be one of" in e for e in errors)


# ── registry_utils ────────────────────────────────────────────────────────────

def test_find_model_locates_known_model():
    from scripts.training.registry_utils import find_model
    assert find_model("k100s2_v1") == "primary"
    assert find_model("transfer_mlp") == "primary"


def test_find_model_returns_none_for_unknown():
    from scripts.training.registry_utils import find_model
    assert find_model("definitely_not_a_registered_model_xyz") is None


def test_require_available_blocks_existing_unless_overwrite():
    from scripts.training.registry_utils import require_available
    assert require_available("k100s2_v1", overwrite=False) == "primary"
    assert require_available("k100s2_v1", overwrite=True) is None
    assert require_available("definitely_not_a_registered_model_xyz", overwrite=False) is None


# ── check_phits_upload ────────────────────────────────────────────────────────

PHITS_HE = REPO / "data" / "raw" / "phits" / "high_energy"


@pytest.mark.skipif(not PHITS_HE.exists(), reason="PHITS data not present")
def test_check_one_passes_for_real_file():
    from scripts.utilities.check_phits_upload import check_one
    sample = next(PHITS_HE.glob("*.out"), None)
    if sample is None:
        pytest.skip("No .out files in HE directory")
    assert check_one(sample) == []


def test_check_one_flags_bad_naming(tmp_path):
    from scripts.utilities.check_phits_upload import check_one
    bad = tmp_path / "NotAValidName.out"
    bad.write_text("garbage content, not a PHITS file")
    problems = check_one(bad)
    assert problems  # non-empty: wrong folder AND unparseable
