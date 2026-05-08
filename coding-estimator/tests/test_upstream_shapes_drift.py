"""Detect drift between our shape-label snapshot and upstream.

Claim:
    The SHA256 of `scripts/label_observation_shapes.py` matches the
    `SNAPSHOT_SHA256` pinned in `_upstream_shapes_snapshot.py`.

Plausible wrong implementations:
    - hash a moving target (e.g. the directory path)
    - silently pass when the upstream file is missing
"""

from __future__ import annotations

import hashlib

import pytest

from coding_estimator.ingest.paths import ledger_root
from coding_estimator.labels._upstream_shapes_snapshot import (
    SNAPSHOT_SHA256,
    UPSTREAM_FILE_RELPATH,
)


def _upstream_sha() -> str:
    full = ledger_root() / UPSTREAM_FILE_RELPATH
    if not full.is_file():
        pytest.skip(f"upstream file not present: {full}")
    return hashlib.sha256(full.read_bytes()).hexdigest()


def test_shape_snapshot_matches_upstream() -> None:
    upstream = _upstream_sha()
    assert upstream == SNAPSHOT_SHA256, (
        f"upstream {UPSTREAM_FILE_RELPATH} has changed (now {upstream}); "
        "audit and re-snapshot _upstream_shapes_snapshot.py"
    )
