"""Tests for run directory creation and metadata."""

import json
from pathlib import Path
import tempfile

from shielding_ml.pipelines.run_manager import (
    make_run_id, create_run_dir, write_run_metadata, complete_run
)


def test_run_id_format():
    rid = make_run_id("inference", "concrete-25")
    parts = rid.split("-")
    assert parts[0] == "inference"
    assert len(parts[1]) == 8   # YYYYMMDD
    assert len(parts[2]) == 6   # HHMMSS


def test_create_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        runs_root = Path(tmp)
        run_dir = create_run_dir(runs_root, "inference", "test")
        assert run_dir.exists()
        assert run_dir.parent.name == "inference"


def test_write_and_complete_run():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "test-run"
        run_dir.mkdir()
        write_run_metadata(run_dir, config={"model": "test"})
        assert (run_dir / "run.json").exists()
        meta = json.loads((run_dir / "run.json").read_text())
        assert meta["status"] == "running"
        complete_run(run_dir)
        meta2 = json.loads((run_dir / "run.json").read_text())
        assert meta2["status"] == "completed"
        assert "finished_at" in meta2
