#!/usr/bin/env python
"""Build the frozen-surface tb_live_v2 process-dynamics audit package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_estimator.eval.process_dynamics import AuditFailure, run_process_dynamics_package


def run(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    manifest_path: Path,
    runs_dir: Path,
    out_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    return run_process_dynamics_package(
        checkpoints_path=checkpoints_path,
        labels_path=labels_path,
        manifest_path=manifest_path,
        runs_dir=runs_dir,
        out_dir=out_dir,
        figures_dir=figures_dir,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/tb_live_v2"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        outputs = run(
            checkpoints_path=args.checkpoints,
            labels_path=args.labels,
            manifest_path=args.manifest,
            runs_dir=args.runs_dir,
            out_dir=args.out_dir,
            figures_dir=args.figures_dir,
        )
    except AuditFailure as exc:
        print(f"audit failure [{exc.verdict}]: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {outputs['progress_audit_md']}")
    print(f"wrote {outputs['validation_audit_md']}")
    print(f"wrote {outputs['case_studies_md']}")
    print(f"wrote {outputs['result_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

