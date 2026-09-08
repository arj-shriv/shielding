"""
Tests for scripts/utilities/source_spectra_editor.py — the pure-text editor
that generates the PR diff for Model Explorer's "Add source spectrum" panel.

Run from the repo root: pytest tests/
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
SOURCE_SPECTRA_PATH = REPO / "src" / "shielding_ml" / "data" / "source_spectra.py"


@pytest.fixture
def original_text() -> str:
    return SOURCE_SPECTRA_PATH.read_text()


def test_insert_new_source_produces_valid_python(original_text):
    from scripts.utilities.source_spectra_editor import insert_new_source
    new_text = insert_new_source(original_text, "concrete100", "Concrete_100cm.out",
                                  10.0, "Concrete 100cm")
    ast.parse(new_text)  # raises SyntaxError if malformed


def test_insert_new_source_adds_definition_and_registry_entry(original_text):
    from scripts.utilities.source_spectra_editor import insert_new_source
    new_text = insert_new_source(original_text, "concrete100", "Concrete_100cm.out",
                                  10.0, "Concrete 100cm")
    assert "concrete100 = _make_noisy_source('Concrete_100cm.out', E_cut=10.0)" in new_text
    assert "'concrete100': concrete100," in new_text


def test_insert_new_source_leaves_existing_entries_intact(original_text):
    from scripts.utilities.source_spectra_editor import insert_new_source
    new_text = insert_new_source(original_text, "concrete100", "Concrete_100cm.out",
                                  10.0, "Concrete 100cm")
    for existing in ("tracknet10", "concrete25", "concrete45", "steel27", "bpe25"):
        assert f"'{existing}':" in new_text and f" {existing},\n" in new_text


def test_insert_new_source_missing_anchor_raises():
    from scripts.utilities.source_spectra_editor import insert_new_source, SourceEditError
    with pytest.raises(SourceEditError):
        insert_new_source("no anchors here at all", "concrete100", "x.out", 10.0, "x")


def test_validate_var_name_accepts_good_name():
    from scripts.utilities.source_spectra_editor import validate_var_name
    validate_var_name("concrete100", {"tracknet10"})  # should not raise


@pytest.mark.parametrize("bad_name,existing", [
    ("Concrete100", set()),
    ("100concrete", set()),
    ("concrete-100", set()),
    ("class", set()),
    ("concrete25", {"concrete25"}),
])
def test_validate_var_name_rejects_bad_names(bad_name, existing):
    from scripts.utilities.source_spectra_editor import validate_var_name, SourceEditError
    with pytest.raises(SourceEditError):
        validate_var_name(bad_name, existing)
