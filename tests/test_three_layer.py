"""
Tests for scripts/inference/three_layer.py's pure logic (no model loading —
that's covered by manual smoke tests against real PHITS data, see
scripts/inference/run_3layer_comparison.py).

Run from the repo root: pytest tests/
"""
from __future__ import annotations

import numpy as np
import pytest


def test_resolve_source_tracknet10_returns_a_spec():
    from scripts.inference.three_layer import _resolve_source
    a_spec = np.full(250, 1 / 250)
    upstream = np.random.rand(250)
    result = _resolve_source("tracknet10", upstream, a_spec)
    assert np.array_equal(result, a_spec)


def test_resolve_source_actual_renormalises_upstream():
    from scripts.inference.three_layer import _resolve_source
    a_spec = np.full(250, 1 / 250)
    upstream = np.arange(1, 251, dtype=float)   # arbitrary, sum != 1
    result = _resolve_source("actual", upstream, a_spec)
    assert result == pytest.approx(upstream / upstream.sum())
    assert result.sum() == pytest.approx(1.0)


def test_resolve_source_rejects_bad_mode():
    from scripts.inference.three_layer import _resolve_source
    with pytest.raises(ValueError):
        _resolve_source("bogus", np.ones(250), np.ones(250))


def test_approach_labels_cover_all_four_combinations():
    from scripts.inference.three_layer import APPROACH_LABELS
    keys = set(APPROACH_LABELS.keys())
    expected = {
        (1, "tracknet10", "tracknet10"),
        (2, "tracknet10", "actual"),
        (3, "actual", "tracknet10"),
        (4, "actual", "actual"),
    }
    assert keys == expected
