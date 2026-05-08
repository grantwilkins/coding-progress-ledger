#!/usr/bin/env python
"""U6 driver — tb_live_v2 exact-task evaluation.

Usage:
    uv run python scripts/run_tb_live_v2_eval.py \
        --checkpoints datasets/checkpoints_tb_live_v2.parquet \
        --labels datasets/labels_tb_live_v2.parquet \
        --manifest datasets/manifests/tb_live_v2.csv \
        --out reports/tb_live_v2_eval.md \
        --shape-profile reports/tb_live_v2_shape_profile.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.harness import cells_to_frame
from coding_estimator.eval.tb_live_v2 import (
    build_tb_live_v2_profile,
    evaluate_tb_live_v2,
    write_tb_live_v2_report,
    write_tb_live_v2_shape_profile,
)
from coding_estimator.io import write_csv


def run(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    manifest_path: Path,
    out_path: Path,
    shape_profile_path: Path,
    metrics_csv_path: Path | None = None,
) -> tuple[Path, Path]:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    manifest_df = pd.read_csv(manifest_path)

    cells = evaluate_tb_live_v2(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    profile = build_tb_live_v2_profile(manifest_df=manifest_df)

    if metrics_csv_path is not None:
        write_csv(
            cells_to_frame(cells),
            metrics_csv_path,
            sort_by=["scheme", "source_slice", "target", "model"],
        )

    write_tb_live_v2_report(out_path, cells=cells, profile=profile)
    write_tb_live_v2_shape_profile(shape_profile_path, profile=profile)
    return out_path, shape_profile_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shape-profile", type=Path, required=True)
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="optional CSV path for all EvalCell rows",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path, shape_profile_path = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        manifest_path=args.manifest,
        out_path=args.out,
        shape_profile_path=args.shape_profile,
        metrics_csv_path=args.metrics_csv,
    )
    print(f"wrote tb_live_v2 eval to {out_path}")
    print(f"wrote tb_live_v2 shape profile to {shape_profile_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
