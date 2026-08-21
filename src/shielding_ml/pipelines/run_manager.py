"""
Run directory management and unique run-ID generation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit(repo_root: Path) -> str:
    """Return current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def make_run_id(run_type: str, short_name: str = "") -> str:
    """
    Generate a unique run ID: <run-type>-<YYYYMMDD>-<HHMMSS>[-<short-name>]
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    parts = [run_type, ts]
    if short_name:
        parts.append(short_name.lower().replace(" ", "-")[:20])
    return "-".join(parts)


def create_run_dir(
    runs_root: Path,
    run_type: str,
    short_name: str = "",
    *,
    exist_ok: bool = False,
) -> Path:
    """
    Create and return a unique run directory under runs_root/<run_type>/<run_id>/.
    Raises FileExistsError if the directory exists and exist_ok=False.
    """
    run_id  = make_run_id(run_type, short_name)
    run_dir = runs_root / run_type / run_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir


def write_run_metadata(
    run_dir: Path,
    config: dict,
    *,
    repo_root: Path | None = None,
) -> None:
    """
    Write run.json into run_dir with timestamp, git hash and resolved config.
    """
    meta = {
        "run_id":    run_dir.name,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root or run_dir),
        "status":    "running",
        "config":    config,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))


def complete_run(run_dir: Path, status: str = "completed") -> None:
    """Update run.json status to completed (or failed)."""
    run_json = run_dir / "run.json"
    if run_json.exists():
        meta = json.loads(run_json.read_text())
        meta["status"] = status
        meta["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
        run_json.write_text(json.dumps(meta, indent=2))
