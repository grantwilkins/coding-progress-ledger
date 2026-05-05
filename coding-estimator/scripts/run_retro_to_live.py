#!/usr/bin/env python
"""Workstream L3 driver — retrospective→live transfer with feature-group
ablation.

Usage:
    uv run python scripts/run_retro_to_live.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out reports/retro_to_live_transfer.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.transfer import evaluate_transfer, write_transfer_report


def run(*, checkpoints_path: Path, labels_path: Path, out_path: Path) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    rows = evaluate_transfer(checkpoints_df=checkpoints_df, labels_df=labels_df)
    write_transfer_report(
        out_path,
        rows,
        summary=(
            "Feature-group ablation drops one of {closure, frontier, instability, "
            "discovery} at a time and retrains. Annotation-leakage caveat from § C1 "
            "applies to retrospective fits."
        ),
    )
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_path=args.out,
    )
    print(f"wrote retro->live transfer report to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
