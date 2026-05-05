#!/usr/bin/env python
"""Run Workstream I model ladder over prepared datasets.

Usage:
    uv run python scripts/run_model_ladder.py \
        --checkpoints datasets/checkpoints_all.parquet \
        --labels datasets/labels_all.parquet \
        --models-dir models \
        --reports-dir reports/model_ladder_i
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.eval.harness import cells_to_frame
from coding_estimator.io import write_csv
from coding_estimator.models.empirical_bin import MODEL_ID as EMPIRICAL_ID
from coding_estimator.models.empirical_bin import train_empirical_bin_bundle
from coding_estimator.models.logreg import MODEL_ID as LOGREG_ID
from coding_estimator.models.logreg import train_logreg_bundle
from coding_estimator.reports.render import write_eval_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoints_df = pd.read_parquet(args.checkpoints)
    labels_df = pd.read_parquet(args.labels)

    empirical_cells = train_empirical_bin_bundle(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        out_dir=args.models_dir / EMPIRICAL_ID,
    )
    logreg_cells = train_logreg_bundle(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        out_dir=args.models_dir / LOGREG_ID,
    )
    cells = empirical_cells + logreg_cells
    frame = cells_to_frame(cells)
    write_csv(
        frame,
        args.reports_dir / "metrics.csv",
        sort_by=["scheme", "source_slice", "target", "model"],
    )
    write_eval_report(
        args.reports_dir / "eval_report.md",
        title="Workstream I model ladder",
        cells=cells,
        summary=(
            "I0 empirical-bin and I1 logistic-regression models evaluated "
            "on the combined holdout split and per-source loro diagnostics. "
            "All models consume prefix-only checkpoint features and write "
            "bundles under `models/`."
        ),
    )
    print(args.models_dir / EMPIRICAL_ID)
    print(args.models_dir / LOGREG_ID)
    print(args.reports_dir / "metrics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
