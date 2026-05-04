"""The D5 audit gate is structural, not advisory.

Workstream G training code calls assert_audit_clean() before fit/
predict. The helper raises AuditNotCleanError when:
- the audit file is missing entirely
- the audit file does not end in 'Overall: PASS'

Claim:
    assert_audit_clean enforces AGENTS.md invariant 8: no training
    starts without a clean audit on disk.

Plausible wrong implementations:
    - silently pass when the audit is missing (defaulting to True)
    - check only the existence of the file, not its verdict
    - look for the substring 'PASS' anywhere in the file (would also
      match a placeholder section that says 'PASS — no leakage')
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.models import AuditNotCleanError, assert_audit_clean


def test_missing_audit_raises(tmp_path: Path) -> None:
    with pytest.raises(AuditNotCleanError, match="not found"):
        assert_audit_clean(tmp_path / "nonexistent.md")


def test_failing_audit_raises(tmp_path: Path) -> None:
    audit = tmp_path / "audit.md"
    audit.write_text(
        "# Audit\n\n## Forbidden columns\nFAIL\n\n---\nOverall: FAIL\n",
        encoding="utf-8",
    )
    with pytest.raises(AuditNotCleanError, match="does not end in"):
        assert_audit_clean(audit)


def test_passing_audit_returns_path(tmp_path: Path) -> None:
    audit = tmp_path / "audit.md"
    audit.write_text(
        "# Audit\n\n## Forbidden\nPASS\n\n---\nOverall: PASS\n",
        encoding="utf-8",
    )
    out = assert_audit_clean(audit)
    assert out == audit


def test_substring_pass_anywhere_does_not_satisfy(tmp_path: Path) -> None:
    """A PASS that is NOT on the final line must not satisfy the gate.
    Otherwise an audit with 'PASS — no leakage' in some middle section
    but 'Overall: FAIL' at the end would falsely satisfy the gate."""
    audit = tmp_path / "audit.md"
    audit.write_text(
        "# Audit\n\n## Forbidden\nPASS\n\n---\nOverall: FAIL\n",
        encoding="utf-8",
    )
    with pytest.raises(AuditNotCleanError):
        assert_audit_clean(audit)


def test_real_built_audit_satisfies_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: build a real audit on tb_live, write it, and confirm
    the gate accepts it."""
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    from coding_estimator.checkpoints.build import build_source_frame
    from coding_estimator.leakage.audit import build_audit, write_audit

    df = build_source_frame("tb_live")
    audit = build_audit(df, sources=["tb_live"])
    out = write_audit(audit, tmp_path / "reports")
    assert assert_audit_clean(out) == out
