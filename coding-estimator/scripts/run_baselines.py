#!/usr/bin/env python
"""H2/H3/H7 driver — run the v0 baseline ladder under loro+ltfo per
source plus loso to tb_live, plus phase × shape slice metrics, and
render the jinja eval report.

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
from coding_estimator.checkpoints.features.registry import all_features
from coding_estimator.eval.harness import (
    EvalCell,
    cells_to_frame,
    evaluate_cell,
    predict_cell,
)
from coding_estimator.eval.slices import (
    SliceCell,
    evaluate_phase_slices,
    evaluate_shape_slices,
    slice_cells_to_frame,
)
from coding_estimator.io import write_csv
from coding_estimator.labels.shapes import shape_rows_for_source
from coding_estimator.profile.budget import compute_budget
from coding_estimator.reports.baselines import (
    write_baseline_calibration_md,
    write_baseline_results_md,
)
from coding_estimator.reports.render import write_eval_report
from coding_estimator.splits.builder import attach_task_family, task_family_map
from coding_estimator.splits.protocol import Fold, Split, loro, ltfo

V0_BINARY_TARGETS = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
    "y_timeout",
)
LOSO_TEST_SOURCE = "tb_live"


def _apply_canonical_fills(checkpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the registry's per-source canonical fill to feature columns
    so the strict-NaN guard in `_features` doesn't trip on cells whose
    declared missingness semantic resolves to 0/False. Cells whose fill
    is None (NOT_APPLICABLE_TO_SOURCE / UNKNOWN_DUE_TO_MISSING_ARTIFACT)
    are left as NaN — `_features` will still hard-fail those, which is
    the contract."""
    df = checkpoints_df.copy()
    feats = {f.column_name: f for f in all_features()}
    for source, sub in df.groupby("source", sort=True):
        idx = sub.index
        for col in df.columns:
            f = feats.get(col)
            if f is None or f.dtype not in ("int", "float", "bool"):
                continue
            fill = f.canonical_fill_for(str(source))
            if fill is None:
                continue
            df.loc[idx, col] = sub[col].fillna(fill)
    return df


def _wide_targets(labels_df: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    sub = labels_df[labels_df["target_name"].isin(targets)].copy()
    sub = sub[~sub["is_masked"].astype(bool)]
    keys = ["run_id", "source", "checkpoint_id", "target_name"]
    dup = sub[sub.duplicated(subset=keys, keep=False)]
    if not dup.empty:
        raise ValueError(
            f"label table contains {len(dup)} duplicated key rows; "
            f"first offender: {dup.iloc[0][keys].to_dict()}"
        )
    return sub.pivot(
        index=["run_id", "source", "checkpoint_id"],
        columns="target_name",
        values="label_value",
    ).reset_index()


def _shapes_lookup(sources: list[str]) -> pd.DataFrame:
    import warnings
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for s in sources:
        rows = shape_rows_for_source(s)
        if rows:
            frames.append(pd.DataFrame(rows))
        else:
            missing.append(s)
    if missing:
        warnings.warn(
            f"shape labels missing for {missing}; shape slice metrics will skip them",
            stacklevel=2,
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _slice_for_cell(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str,
    spec,
    split: Split,
    source_slice: str,
    sources_in_train: tuple[str, ...],
    shapes_df: pd.DataFrame,
) -> list[SliceCell]:
    preds = predict_cell(
        checkpoints_df=checkpoints_df, labels_df=labels_df,
        target=target, spec=spec, split=split, sources_in_train=sources_in_train,
    )
    if preds.empty:
        return []
    out = evaluate_phase_slices(
        preds, target=target, model=spec.name, scheme=split.scheme, source_slice=source_slice,
    )
    out.extend(evaluate_shape_slices(
        preds, shapes_df,
        target=target, model=spec.name, scheme=split.scheme, source_slice=source_slice,
    ))
    return out


def _per_source_within(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    wide: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> tuple[list[EvalCell], list[SliceCell]]:
    cells: list[EvalCell] = []
    slices: list[SliceCell] = []
    sources = sorted(checkpoints_df["source"].unique())
    for source in sources:
        ck_src = checkpoints_df[checkpoints_df["source"] == source]
        wide_src = wide[wide["source"] == source]
        if ck_src.empty or wide_src.empty:
            continue
        family_map = task_family_map(source)
        # `compute_budget` for ltfo groups by a column named
        # `task_family`; for tb_live_v2 the attached values are exact
        # task_ids so same-task cross-arm replications stay in one fold.
        wide_for_budget = wide_src.assign(task_family=wide_src["run_id"].map(family_map))
        budget = {
            (c.target, c.split_scheme): c
            for c in compute_budget(
                wide_for_budget, targets=V0_BINARY_TARGETS, schemes=("loro", "ltfo")
            )
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
        loro_split = loro(ck_src)

        ltfo_split: Split | None = None
        if any(v is not None for v in family_map.values()):
            ck_with_fam = attach_task_family(ck_src, family_map).dropna(subset=["task_family"])
            if ck_with_fam["task_family"].nunique() >= 2:
                ltfo_split = ltfo(ck_with_fam)

        for scheme_name, split in (("loro", loro_split), ("ltfo", ltfo_split)):
            if split is None:
                for target in V0_BINARY_TARGETS:
                    for spec in V0_BASELINES:
                        cells.append(EvalCell(
                            target=target, model=spec.name, scheme=scheme_name,
                            source_slice=source, feasible=False,
                            n_runs_train=None, n_runs_test=None, n_checkpoints_test=None,
                            positive_rate_data=None, predicted_positive_rate=None,
                            auroc=None, brier=None, log_loss=None, ece=None,
                            brier_ci_low=None, brier_ci_high=None,
                            note="ltfo: no task_family on source",
                        ))
                continue
            for target in V0_BINARY_TARGETS:
                cell_budget = budget.get((target, scheme_name))
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
                    if feasible:
                        slices.extend(_slice_for_cell(
                            checkpoints_df=ck_src, labels_df=labels_df,
                            target=target, spec=spec, split=split,
                            source_slice=source, sources_in_train=(source,),
                            shapes_df=shapes_df,
                        ))
    return cells, slices


def _loso_to_tb_live(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
) -> tuple[list[EvalCell], list[SliceCell]]:
    sources = sorted(checkpoints_df["source"].unique())
    if LOSO_TEST_SOURCE not in sources or len(sources) < 2:
        return [], []
    test_runs = tuple(
        sorted(
            checkpoints_df.loc[
                checkpoints_df["source"] == LOSO_TEST_SOURCE, "run_id"
            ].unique()
        )
    )
    train_runs = tuple(
        sorted(
            checkpoints_df.loc[
                checkpoints_df["source"] != LOSO_TEST_SOURCE, "run_id"
            ].unique()
        )
    )
    train_sources = tuple(sorted(s for s in sources if s != LOSO_TEST_SOURCE))
    if not test_runs or not train_runs:
        return [], []
    fold = Fold(
        fold_id=f"loso::{LOSO_TEST_SOURCE}",
        train_run_ids=train_runs,
        test_run_ids=test_runs,
    )
    split = Split(scheme="loso", seed=0, folds=(fold,))
    cells: list[EvalCell] = []
    slices: list[SliceCell] = []
    source_slice = f"loso->{LOSO_TEST_SOURCE}"
    for target in V0_BINARY_TARGETS:
        for spec in V0_BASELINES:
            cell = evaluate_cell(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                target=target,
                spec=spec,
                split=split,
                source_slice=source_slice,
                sources_in_train=train_sources,
                feasible=True,
            )
            cells.append(cell)
            if cell.feasible:
                slices.extend(_slice_for_cell(
                    checkpoints_df=checkpoints_df, labels_df=labels_df,
                    target=target, spec=spec, split=split,
                    source_slice=source_slice, sources_in_train=train_sources,
                    shapes_df=shapes_df,
                ))
    return cells, slices


def run(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    out_dir: Path,
) -> Path:
    checkpoints_df = _apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    wide = _wide_targets(labels_df, V0_BINARY_TARGETS)
    shapes_df = _shapes_lookup(sorted(checkpoints_df["source"].unique()))

    within_cells, within_slices = _per_source_within(checkpoints_df, labels_df, wide, shapes_df)
    cross_cells, cross_slices = _loso_to_tb_live(checkpoints_df, labels_df, shapes_df)

    cells = within_cells + cross_cells
    slices = within_slices + cross_slices

    out_dir.mkdir(parents=True, exist_ok=True)
    frame = cells_to_frame(cells)
    csv_path = write_csv(
        frame,
        out_dir / "baseline_metrics.csv",
        sort_by=["scheme", "source_slice", "target", "model"],
    )
    write_baseline_results_md(frame, out_dir / "baseline_results.md")
    write_baseline_calibration_md(frame, out_dir / "baseline_calibration.md")

    if slices:
        slice_frame = slice_cells_to_frame(slices)
        write_csv(
            slice_frame,
            out_dir / "baseline_slice_metrics.csv",
            sort_by=["scheme", "source_slice", "target", "model", "slice_kind", "slice_value"],
        )

    write_eval_report(
        out_dir / "eval_report.md",
        title="Baseline ladder — v0 eval report",
        cells=cells,
        slices=slices,
        summary=(
            "v0 baselines (G1 constant, G2 time-only, G4 ledger-basic) "
            "evaluated under loro/ltfo per source and loso to tb_live. "
            "Run-level bootstrap CIs (B=1000, seed=0). "
            "Slices flagged with < 5 positives or < 5 negatives are emitted "
            "as `n/a (insufficient data)`."
        ),
    )
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
