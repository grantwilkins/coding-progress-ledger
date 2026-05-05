#!/usr/bin/env python
"""Workstream O driver — O1, O5, O7 adversarial failure-mode tests.

Usage:
    uv run python scripts/run_failure_modes.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports/failure_modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.failure_modes import (
    HEADLINE_TARGET,
    TB_LIVE,
    all_results_json,
    evaluate_o1,
    evaluate_o5,
    evaluate_o7,
    write_failure_mode_report,
)
from coding_estimator.eval.harness import predict_cell
from coding_estimator.io import write_json
from coding_estimator.splits.protocol import loro


def _final_success_map(labels_df: pd.DataFrame) -> dict[str, int]:
    s = labels_df[
        (labels_df["target_name"] == HEADLINE_TARGET)
        & (~labels_df["is_masked"].astype(bool))
    ].drop_duplicates("run_id")
    return {str(r["run_id"]): int(r["label_value"]) for _, r in s.iterrows()}


def _g4_per_source_predictions(
    *, checkpoints_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Concatenated G4 LORO predictions across every source for the
    headline target. Used as the prediction frame for O1."""
    pieces: list[pd.DataFrame] = []
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        preds = predict_cell(
            checkpoints_df=sub,
            labels_df=labels_df,
            target=HEADLINE_TARGET,
            spec=LEDGER_BASIC,
            split=split,
            sources_in_train=(source,),
        )
        if not preds.empty:
            pieces.append(preds)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def run(*, checkpoints_path: Path, labels_path: Path, out_dir: Path) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_success = _final_success_map(labels_df)
    g4_preds = _g4_per_source_predictions(
        checkpoints_df=checkpoints_df, labels_df=labels_df
    )

    o1 = evaluate_o1(
        predictions_df=g4_preds,
        checkpoints_df=checkpoints_df,
        final_success=final_success,
    )
    o5 = evaluate_o5(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        test_source=TB_LIVE,
        target=HEADLINE_TARGET,
    )
    o7 = evaluate_o7(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        target=HEADLINE_TARGET,
    )

    md_path = write_failure_mode_report(
        out_dir / "failure_modes.md",
        o1=o1, o5=o5, o7=o7,
        summary=(
            "Adversarial tests against the v0 G4 (`ledger_basic`) "
            "estimator. O1, O5, O7 only — O2/O3/O4/O6 are deferred at "
            "current N (see TASKS.md § Workstream O)."
        ),
    )
    write_json(all_results_json(o1, o5, o7), out_dir / "failure_modes.json")
    return md_path


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
    print(f"wrote failure-mode tests to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
