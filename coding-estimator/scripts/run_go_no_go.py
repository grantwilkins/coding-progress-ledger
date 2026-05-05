#!/usr/bin/env python
"""Workstream P1 driver — render `reports/ESTIMATOR_GO_NO_GO.md`.

Usage:
    uv run python scripts/run_go_no_go.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out reports/ESTIMATOR_GO_NO_GO.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.go_no_go import (
    evaluate_gate,
    write_gate_report,
)
from coding_estimator.io import write_json


def run(*, checkpoints_path: Path, labels_path: Path, out_path: Path,
        d5_audit_path: Path | None = None) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    report = evaluate_gate(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        d5_audit_path=d5_audit_path,
    )
    write_gate_report(
        out_path,
        report,
        summary=(
            "Eight conditions from TASKS.md § Workstream P. The gate is "
            "intentionally a *no-regression* gate at v0 — see § P-future "
            "for the aspirational gate that requires CI exclusion on "
            "tb_live and ECE within plan."
        ),
    )
    write_json(report.to_dict(), out_path.with_suffix(".json"))
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--d5-audit",
        type=Path,
        default=Path("reports/d5_audit.json"),
        help="path to D5 audit JSON (default: reports/d5_audit.json)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_path=args.out,
        d5_audit_path=args.d5_audit,
    )
    print(f"wrote go/no-go gate report to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
