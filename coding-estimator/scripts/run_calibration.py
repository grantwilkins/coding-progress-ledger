#!/usr/bin/env python
"""Workstream J driver — render reliability, slice, and headline
calibration reports for the v0 baselines.

For each canonical source × baseline (G2 time-only, G4 ledger-basic),
generate cross-validated test predictions under per-source LORO and
under LOSO -> tb_live, then emit:

    reports/calibration/calibration_<model>_<source>.md   (J2)
    reports/calibration/calibration_slices.md             (J3)
    reports/calibration/calibration_v0.md                 (J5)

Usage:
    uv run python scripts/run_calibration.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports/calibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, TIME_ONLY, BaselineSpec
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.calibration.report import (
    HeadlineRow,
    SliceCalibrationRow,
    headline_rows,
    slice_calibration_rows,
    write_headline_report,
    write_reliability_report,
    write_slice_report,
)
from coding_estimator.eval.harness import predict_cell
from coding_estimator.labels.shapes import shape_rows_for_source
from coding_estimator.splits.protocol import Fold, Split, loro

V0_TARGETS_FOR_CALIBRATION: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
)
V0_MODELS_FOR_CALIBRATION: tuple[BaselineSpec, ...] = (TIME_ONLY, LEDGER_BASIC)
LOSO_TEST_SOURCE = "tb_live"


def _shapes_lookup(sources: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for s in sources:
        rows = shape_rows_for_source(s)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _per_source_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    source: str,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Returns {(model_name, target): predictions_df} for one source
    under per-source LORO."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    sub = checkpoints_df[checkpoints_df["source"] == source]
    if sub["run_id"].nunique() < 2:
        return out
    split = loro(sub)
    for spec in V0_MODELS_FOR_CALIBRATION:
        for target in V0_TARGETS_FOR_CALIBRATION:
            preds = predict_cell(
                checkpoints_df=sub,
                labels_df=labels_df,
                target=target,
                spec=spec,
                split=split,
                sources_in_train=(source,),
            )
            if not preds.empty:
                out[(spec.name, target)] = preds
    return out


def _loso_predictions(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Returns {(model_name, target): predictions_df} for the
    LOSO -> tb_live transfer split."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    sources = sorted(checkpoints_df["source"].unique())
    if LOSO_TEST_SOURCE not in sources or len(sources) < 2:
        return out
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
    train_sources = tuple(s for s in sources if s != LOSO_TEST_SOURCE)
    if not test_runs or not train_runs:
        return out
    fold = Fold(
        fold_id=f"loso::{LOSO_TEST_SOURCE}",
        train_run_ids=train_runs,
        test_run_ids=test_runs,
    )
    split = Split(scheme="loso", seed=0, folds=(fold,))
    for spec in V0_MODELS_FOR_CALIBRATION:
        for target in V0_TARGETS_FOR_CALIBRATION:
            preds = predict_cell(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                target=target,
                spec=spec,
                split=split,
                sources_in_train=train_sources,
            )
            if not preds.empty:
                out[(spec.name, target)] = preds
    return out


def run(*, checkpoints_path: Path, labels_path: Path, out_dir: Path, n_bins: int = 10) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    sources = sorted(checkpoints_df["source"].unique())
    shapes_df = _shapes_lookup(sources)

    out_dir.mkdir(parents=True, exist_ok=True)

    all_slice_rows: list[SliceCalibrationRow] = []
    all_headline: list[HeadlineRow] = []

    # Per-source LORO
    for source in sources:
        per_source = _per_source_predictions(
            checkpoints_df=checkpoints_df,
            labels_df=labels_df,
            source=source,
        )
        if not per_source:
            continue
        per_model: dict[str, dict[str, pd.DataFrame]] = {}
        for (model_name, target), preds in per_source.items():
            per_model.setdefault(model_name, {})[target] = preds
        for model_name, by_target in per_model.items():
            write_reliability_report(
                out_dir / f"calibration_{model_name}_{source}.md",
                title=f"Calibration — {model_name} on {source} (LORO)",
                model=model_name,
                source=source,
                by_target=by_target,
                n_bins=n_bins,
            )
            for target, preds in by_target.items():
                all_slice_rows.extend(
                    slice_calibration_rows(
                        model=model_name,
                        source=source,
                        target=target,
                        predictions_df=preds,
                        checkpoints_df=checkpoints_df,
                        shapes_df=shapes_df if not shapes_df.empty else None,
                        target_horizon=str(_horizon_for(target)),
                        n_bins=n_bins,
                    )
                )
                all_headline.append(
                    headline_rows(
                        [(model_name, source, target, preds)],
                        n_bins=n_bins,
                    )[0]
                )

    # LOSO -> tb_live
    loso = _loso_predictions(checkpoints_df=checkpoints_df, labels_df=labels_df)
    if loso:
        per_model: dict[str, dict[str, pd.DataFrame]] = {}
        for (model_name, target), preds in loso.items():
            per_model.setdefault(model_name, {})[target] = preds
        loso_label = f"loso->{LOSO_TEST_SOURCE}"
        for model_name, by_target in per_model.items():
            write_reliability_report(
                out_dir / f"calibration_{model_name}_{loso_label}.md",
                title=f"Calibration — {model_name} on {LOSO_TEST_SOURCE} (LOSO)",
                model=model_name,
                source=loso_label,
                by_target=by_target,
                n_bins=n_bins,
            )
            for target, preds in by_target.items():
                all_slice_rows.extend(
                    slice_calibration_rows(
                        model=model_name,
                        source=loso_label,
                        target=target,
                        predictions_df=preds,
                        checkpoints_df=checkpoints_df,
                        shapes_df=shapes_df if not shapes_df.empty else None,
                        target_horizon=str(_horizon_for(target)),
                        n_bins=n_bins,
                    )
                )
                all_headline.append(
                    headline_rows(
                        [(model_name, loso_label, target, preds)],
                        n_bins=n_bins,
                    )[0]
                )

    slice_path = write_slice_report(
        out_dir / "calibration_slices.md",
        title="Calibration slices — v0",
        rows=all_slice_rows,
        summary=(
            "Per-(model, source, target) calibration across J3 slice axes: "
            "source, target_horizon, phase (early/middle/late), shape class, "
            "progress bucket, validation state."
        ),
    )

    headline_path = write_headline_report(
        out_dir / "calibration_v0.md",
        title="Calibration — v0 headline",
        rows=all_headline,
        summary=(
            "Headline rollup for the v0 calibration gate. Cross-validated "
            "isotonic recalibration uses K-fold over run_ids."
        ),
    )

    return headline_path


def _horizon_for(target: str) -> str | None:
    from coding_estimator.labels.registry import V0_TARGETS

    meta = V0_TARGETS.get(target)
    if meta is None:
        return None
    if meta.horizon_value is None:
        return f"{meta.horizon_units}/none"
    return f"{meta.horizon_units}/{meta.horizon_value}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        out_dir=args.out_dir,
        n_bins=args.n_bins,
    )
    print(f"wrote calibration headline to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
