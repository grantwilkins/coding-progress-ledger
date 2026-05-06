"""S2 — audit a captured workflow directory and emit per_layer byte sums.

Usage:
    uv run python scripts/run_s2_audit.py <workflow_dir> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from agent_migrate_agent.state_layers import audit_workflow_directory, write_audit_artifacts


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: run_s2_audit.py <workflow_dir> <out_dir>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    workflow_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    report = audit_workflow_directory(workflow_dir)
    write_audit_artifacts(report, out_dir)
    print(
        f"audited {report.workflow_dir}: "
        f"total={report.total_bytes:,} bytes, "
        f"must_move={report.must_move_bytes:,} bytes, "
        f"skipped_symlinks={report.skipped_symlinks}"
    )


if __name__ == "__main__":
    main()
