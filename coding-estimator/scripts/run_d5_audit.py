#!/usr/bin/env python
"""D5 driver — produce the structured behavioral leakage audit artifact.

Usage:
    uv run python scripts/run_d5_audit.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-json reports/d5_audit.json \\
        --out-md   reports/d5_audit.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.leakage.d5_audit import (
    run_d5_audit,
    write_d5_audit,
    write_d5_summary_md,
)


def run(
    *, checkpoints_path: Path, labels_path: Path,
    out_json: Path, out_md: Path,
    sample_runs: int = 4,
) -> tuple[Path, Path]:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    audit = run_d5_audit(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        sample_runs_for_truncation=sample_runs,
    )
    write_d5_audit(audit, out_json)
    write_d5_summary_md(audit, out_md)
    print(f"D5 clean={audit.clean} findings={len(audit.findings)}")
    return out_json, out_md


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--sample-runs", type=int, default=4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_json, out_md = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_json=args.out_json,
        out_md=args.out_md,
        sample_runs=args.sample_runs,
    )
    print(f"json: {out_json}")
    print(f"md:   {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
