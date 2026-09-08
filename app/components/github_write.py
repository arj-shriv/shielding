"""
GitHub REST API helper for Model Explorer's write actions — PHITS upload, add a
new source spectrum, push a newly trained model.

Every write lands as a branch + pull request, never a direct push to main, so a
human reviews it before it's live (matches the "PR reviewed and merged by a
collaborator" access-tier design in PROPOSAL.txt).

Requires GITHUB_PAT in .streamlit/secrets.toml — a fine-grained, repo-scoped
token with Contents (read/write) and Pull requests (read/write) permissions on
this repo only. GITHUB_REPO / GITHUB_BASE_BRANCH default to this repo's main.

Uses the Contents API (base64 body, works up to 100MB per file — every model
.pkl in this repo is well under that) rather than the lower-level Git Data
(blob/tree/commit) API, since nothing here needs multi-file atomic commits
across a single tree — one Contents API PUT per file, same branch, is simpler
and just as correct for this use case.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone

import requests
import streamlit as st

API = "https://api.github.com"
_TIMEOUT = 30
_TIMEOUT_UPLOAD = 120   # model .pkl pushes can be several MB


def github_write_available() -> bool:
    return bool(st.secrets.get("GITHUB_PAT"))


def _headers() -> dict:
    token = st.secrets.get("GITHUB_PAT")
    if not token:
        raise RuntimeError(
            "GITHUB_PAT is not set in .streamlit/secrets.toml — see "
            ".streamlit/secrets.toml.example. Write actions are unavailable without it."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo() -> str:
    return st.secrets.get("GITHUB_REPO", "arj-shriv/shielding")


def _base_branch() -> str:
    return st.secrets.get("GITHUB_BASE_BRANCH", "main")


def new_branch_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def create_branch(branch_name: str) -> None:
    """Branch off the current tip of the base branch. No-op if it already exists."""
    r = requests.get(f"{API}/repos/{_repo()}/git/ref/heads/{_base_branch()}",
                      headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    base_sha = r.json()["object"]["sha"]

    r = requests.post(f"{API}/repos/{_repo()}/git/refs", headers=_headers(),
                       json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
                       timeout=_TIMEOUT)
    if r.status_code == 422 and "already exists" in r.text:
        return
    r.raise_for_status()


def _get_file_sha(path: str, ref: str) -> str | None:
    """Blob SHA of an existing file on `ref`, or None if it doesn't exist there."""
    r = requests.get(f"{API}/repos/{_repo()}/contents/{path}",
                      headers=_headers(), params={"ref": ref}, timeout=_TIMEOUT)
    if r.status_code == 200:
        return r.json()["sha"]
    if r.status_code == 404:
        return None
    r.raise_for_status()


def get_file_text(path: str, ref: str | None = None) -> str:
    r = requests.get(f"{API}/repos/{_repo()}/contents/{path}",
                      headers=_headers(), params={"ref": ref or _base_branch()},
                      timeout=_TIMEOUT)
    r.raise_for_status()
    return base64.b64decode(r.json()["content"]).decode("utf-8")


def put_file(branch: str, path: str, content_bytes: bytes, message: str) -> None:
    """Create or update one file on `branch` via the Contents API."""
    sha = _get_file_sha(path, ref=branch)
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{API}/repos/{_repo()}/contents/{path}", headers=_headers(),
                      json=payload, timeout=_TIMEOUT_UPLOAD)
    r.raise_for_status()


def open_pull_request(branch: str, title: str, body: str) -> str:
    """Returns the PR's HTML URL."""
    r = requests.post(f"{API}/repos/{_repo()}/pulls", headers=_headers(),
                       json={"title": title, "head": branch, "base": _base_branch(),
                             "body": body},
                       timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()["html_url"]
