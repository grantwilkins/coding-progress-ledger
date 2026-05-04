"""Pin upstream commit and artifact digests so dataset builds are reproducible."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coding_estimator.ingest.paths import ledger_root
from coding_estimator.io import write_json


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def upstream_commit_sha(allow_dirty: bool = False) -> tuple[str, bool]:
    root = ledger_root()
    sha = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain"], root) != ""
    if dirty and not allow_dirty:
        raise RuntimeError(
            f"upstream ledger checkout at {root} is dirty; pass allow_dirty=True "
            "to override (NOT recommended for shipped artifacts)."
        )
    return sha, dirty


def capture_pin(
    pinned_artifacts: dict[str, Path],
    output_path: Path,
    *,
    allow_dirty: bool = False,
) -> Path:
    """Write a pinning manifest. `pinned_artifacts` maps a logical key
    (e.g. "w3_table") to a path (must live under the upstream ledger root)."""
    root = ledger_root()
    sha, dirty = upstream_commit_sha(allow_dirty=allow_dirty)
    manifest: dict[str, Any] = {
        "ledger_repo_path": str(root),
        "ledger_commit_sha": sha,
        "ledger_dirty": dirty,
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": {},
    }
    for key, path in pinned_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"pinned artifact missing: {path}")
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"pinned artifact {path} is not under ledger_root {root}"
            ) from exc
        manifest["artifacts"][key] = {
            "path": str(rel),
            "sha256": _sha256(path),
        }
    return write_json(manifest, output_path)
