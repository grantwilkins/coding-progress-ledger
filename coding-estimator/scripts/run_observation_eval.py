#!/usr/bin/env python
"""Run the observation-upgrade evaluation on tb_live_v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.build import apply_canonical_fills
from coding_estimator.eval.harness import cells_to_frame
from coding_estimator.eval.observation_upgrade import (
    build_profile,
    build_success_diagnostic_cells,
    build_success_slice_summaries,
    evaluate_observation_upgrade,
    write_observation_upgrade_report,
)
from coding_estimator.io import write_csv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoints_df = apply_canonical_fills(pd.read_parquet(args.checkpoints))
    labels_df = pd.read_parquet(args.labels)
    manifest_df = pd.read_csv(args.manifest)

    cells = evaluate_observation_upgrade(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    slices = build_success_slice_summaries(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    diagnostics = build_success_diagnostic_cells(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        manifest_df=manifest_df,
    )
    profile = build_profile(manifest_df)

    write_csv(
        cells_to_frame(cells),
        args.metrics_csv,
        sort_by=["scheme", "source_slice", "target", "model"],
    )
    write_observation_upgrade_report(
        args.out,
        cells=cells,
        profile=profile,
        slices=slices,
        diagnostics=diagnostics,
    )
    print(args.out)
    print(args.metrics_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
