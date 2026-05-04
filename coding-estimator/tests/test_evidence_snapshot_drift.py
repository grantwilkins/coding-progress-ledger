"""Detect drift between our evidence-classification snapshot and the
upstream source. The snapshot is hand-copied; if upstream changes
classify_evidence semantics or its constants, our classifier will
silently disagree until this test fails.

Claim:
    The SHA256 of the upstream rescore_suite_by_category.py file
    matches the SNAPSHOT_SHA256 pinned in
    _upstream_evidence_snapshot.py. When upstream changes, this test
    fails and forces a deliberate re-snapshot.

Plausible wrong implementations:
    - hash a moving target (e.g. the tested directory path)
    - silently pass when the upstream file is missing
"""

from __future__ import annotations

import hashlib

import pytest

from coding_estimator.checkpoints.features._upstream_evidence_snapshot import (
    SNAPSHOT_SHA256,
)
from coding_estimator.ingest.paths import ledger_root


def _upstream_sha() -> str:
    """SHA256 of the upstream classify_evidence source file."""
    full = ledger_root() / "scripts" / "rescore_suite_by_category.py"
    if not full.is_file():
        pytest.skip(f"upstream file not present: {full}")
    return hashlib.sha256(full.read_bytes()).hexdigest()


def test_evidence_snapshot_matches_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """If this test fails, upstream rescore_suite_by_category.py has
    changed. Inspect the diff; if `classify_evidence` semantics were
    modified, update _upstream_evidence_snapshot.py to match (and bump
    SNAPSHOT_SHA256). If only unrelated functions changed, you can
    just bump SNAPSHOT_SHA256 -- but document the bump in the commit."""
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    actual = _upstream_sha()
    assert actual == SNAPSHOT_SHA256, (
        f"upstream evidence file SHA256 ({actual}) does not match snapshot "
        f"({SNAPSHOT_SHA256}); review the diff and update the snapshot if "
        "classify_evidence semantics changed"
    )


def test_snapshot_classify_matches_upstream_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-check the OUTPUTS, not just the file hash. Compare our
    snapshot's classify_evidence against the live upstream function
    on a battery of inputs. This catches the case where a SHA-bump
    accidentally lets a real semantic change slip through."""
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    import importlib.util
    import sys

    upstream_path = ledger_root() / "scripts" / "rescore_suite_by_category.py"
    if not upstream_path.is_file():
        pytest.skip(f"upstream file not present: {upstream_path}")
    spec = importlib.util.spec_from_file_location(
        "_upstream_rescore", upstream_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from coding_estimator.checkpoints.features._upstream_evidence_snapshot import (
        classify_evidence,
    )

    cases = [
        ["pytest passed in 0.3s"],
        ["I think this works"],
        ["edited src/foo.py"],
        ["task.md describes the contract"],
        ["search_file foo, then submit"],
        ["unit test green"],
        [],
        ["", "blank"],
    ]
    for evidence in cases:
        ours = classify_evidence([str(x) for x in evidence])
        theirs = module.classify_evidence([str(x) for x in evidence])
        assert ours == theirs, (evidence, ours, theirs)
