"""Models package — Workstream G entry point.

By design, training code MUST consult the checkpoint-construction audit
before running. The `assert_audit_clean` helper reads the audit report
emitted by D5 and raises if it does not exist or reports failure.

Training scripts should call `assert_audit_clean()` at the start of any
fit/predict path. This makes AGENTS.md invariant 8 structural rather
than advisory.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_AUDIT_PATH = Path("reports") / "CHECKPOINT_CONSTRUCTION_AUDIT.md"


class AuditNotCleanError(RuntimeError):
    """Raised when training is attempted without a passing audit on disk."""


def assert_audit_clean(audit_path: Path | None = None) -> Path:
    """Verify the audit at `audit_path` exists and ends in `Overall: PASS`.
    Returns the path on success; raises AuditNotCleanError otherwise.

    Defaults to `reports/CHECKPOINT_CONSTRUCTION_AUDIT.md` relative to
    the current working directory.
    """
    path = audit_path if audit_path is not None else DEFAULT_AUDIT_PATH
    if not path.is_file():
        raise AuditNotCleanError(
            f"checkpoint construction audit not found at {path}; run "
            "`uv run python scripts/build_checkpoints.py` and emit the "
            "audit before training."
        )
    text = path.read_text(encoding="utf-8")
    last_line = text.rstrip().splitlines()[-1] if text.rstrip() else ""
    if not last_line.strip().endswith("PASS"):
        raise AuditNotCleanError(
            f"checkpoint construction audit at {path} does not end in "
            f"'Overall: PASS' (last line: {last_line!r})"
        )
    return path
