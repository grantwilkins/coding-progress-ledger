#!/usr/bin/env python
"""G7 driver — run the v0 baseline ladder under loro per source plus
loso to tb_live, emit a metrics frame and the markdown reports.

Usage:
    uv run python scripts/run_baselines.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import V0_BASELINES
from coding_estimator.eval.harness import EvalCell, cells_to_frame, evaluate_cell
from coding_estimator.io import write_csv
from coding_estimator.profile.budget import compute_budget
from coding_estimator.reports.baselines import (
    write_baseline_calibration_md,
    write_baseline_results_md,
)
from coding_estimator.splits.protocol import Fold, Split, loro

V0_BINARY_TARGETS = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
    "y_timeout",
)
LOSO_TEST_SOURCE = "tb_live"


def _wide_targets(labels_df: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    """Pivot the long labels frame to wide so the budget feasibility
    helper can read targets as columns. Hard-fails on duplicate
    (run_id, checkpoint_id, target_name) keys — that would mean the
    label builder emitted two distinct values for the same cell."""
    sub = labels_df[labels_df["target_name"].isin(targets)].copy()
    sub = sub[~sub["is_masked"].astype(bool)]
    keys = ["run_id", "source", "checkpoint_id", "target_name"]
    dup = sub[sub.duplicated(subset=keys, keep=False)]
    if not dup.empty:
        raise ValueError(
            f"label table contains {len(dup)} duplicated key rows; "
            f"first offender: {dup.iloc[0][keys].to_dict()}"
        )
    wide = sub.pivot(
        index=["run_id", "source", "checkpoint_id"],
        columns="target_name",
        values="label_value",
    ).reset_index()
    return wide


def _per_source_loro(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    wide: pd.DataFrame,
) -> list[EvalCell]:
    cells: list[EvalCell] = []
    sources = sorted(checkpoints_df["source"].unique())
    for source in sources:
        ck_src = checkpoints_df[checkpoints_df["source"] == source]
        wide_src = wide[wide["source"] == source]
        if ck_src.empty or wide_src.empty:
            continue
        budget = {
            (c.target, c.split_scheme): c
            for c in compute_budget(wide_src, targets=V0_BINARY_TARGETS, schemes=("loro",))
        }
        if ck_src["run_id"].nunique() < 2:
            for target in V0_BINARY_TARGETS:
                for spec in V0_BASELINES:
                    cells.append(EvalCell(
                        target=target, model=spec.name, scheme="loro",
                        source_slice=source, feasible=False,
                        n_runs_train=None, n_runs_test=None, n_checkpoints_test=None,
                        positive_rate_data=None, predicted_positive_rate=None,
                        auroc=None, brier=None, log_loss=None, ece=None,
                        brier_ci_low=None, brier_ci_high=None,
                        note="single-run source",
                    ))
            continue
        split = loro(ck_src)
        for target in V0_BINARY_TARGETS:
            cell_budget = budget.get((target, "loro"))
            feasible = cell_budget is not None and cell_budget.feasible
            for spec in V0_BASELINES:
                cells.append(evaluate_cell(
                    checkpoints_df=ck_src,
                    labels_df=labels_df,
                    target=target,
                    spec=spec,
                    split=split,
                    source_slice=source,
                    sources_in_train=(source,),
                    feasible=feasible,
                ))
    return cells


def _loso_to_tb_live(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> list[EvalCell]:
    sources = sorted(checkpoints_df["source"].unique())
    if LOSO_TEST_SOURCE not in sources or len(sources) < 2:
        return []
    test_runs = tuple(sorted(
        checkpoints_df.loc[checkpoints_df["source"] == LOSO_TEST_SOURCE, "run_id"].unique()
    ))
    train_runs = tuple(sorted(
        checkpoints_df.loc[checkpoints_df["source"] != LOSO_TEST_SOURCE, "run_id"].unique()
    ))
    train_sources = tuple(sorted(s for s in sources if s != LOSO_TEST_SOURCE))
    if not test_runs or not train_runs:
        return []
    fold = Fold(fold_id=f"loso::{LOSO_TEST_SOURCE}", train_run_ids=train_runs, test_run_ids=test_runs)
    split = Split(scheme="loso", seed=0, folds=(fold,))
    cells: list[EvalCell] = []
    for target in V0_BINARY_TARGETS:
        for spec in V0_BASELINES:
            cells.append(evaluate_cell(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                target=target,
                spec=spec,
                split=split,
                source_slice=f"loso->{LOSO_TEST_SOURCE}",
                sources_in_train=train_sources,
                feasible=True,
            ))
    return cells


def run(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    out_dir: Path,
) -> Path:
    checkpoints_df = pd.read_parquet(checkpoints_path)
    labels_df = pd.read_parquet(labels_path)
    wide = _wide_targets(labels_df, V0_BINARY_TARGETS)
    cells = _per_source_loro(checkpoints_df, labels_df, wide)
    cells.extend(_loso_to_tb_live(checkpoints_df, labels_df))
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = cells_to_frame(cells)
    csv_path = write_csv(
        frame,
        out_dir / "baseline_metrics.csv",
        sort_by=["scheme", "source_slice", "target", "model"],
    )
    write_baseline_results_md(frame, out_dir / "baseline_results.md")
    write_baseline_calibration_md(frame, out_dir / "baseline_calibration.md")
    return csv_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_dir=args.out_dir,
    )
    print(f"wrote baseline metrics to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
