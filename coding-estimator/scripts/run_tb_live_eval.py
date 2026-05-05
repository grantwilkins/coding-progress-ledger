#!/usr/bin/env python
"""Workstream K driver — TB-live-only checkpoint eval (K1) and TB
qualitative rollup (K3).

Usage:
    uv run python scripts/run_tb_live_eval.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --out-dir reports/tb_live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC
from coding_estimator.eval.harness import cells_to_frame, predict_cell
from coding_estimator.eval.tb_live import evaluate_tb_only, write_tb_only_report
from coding_estimator.eval.tb_qualitative import (
    TB_LIVE,
    build_rollups,
    write_qualitative_report,
)
from coding_estimator.io import write_csv
from coding_estimator.labels.shapes import shape_rows_for_source
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.splits.protocol import loro


def _final_success_map(labels_df: pd.DataFrame) -> dict[str, int]:
    s = labels_df[
        (labels_df["target_name"] == "y_success_eventual")
        & (labels_df["source"] == TB_LIVE)
        & (~labels_df["is_masked"].astype(bool))
    ]
    if s.empty:
        return {}
    g = s.drop_duplicates("run_id")[["run_id", "label_value"]]
    return {str(r["run_id"]): int(r["label_value"]) for _, r in g.iterrows()}


def run(*, checkpoints_path: Path, labels_path: Path, out_dir: Path) -> Path:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # K1
    cells, raw_cells = evaluate_tb_only(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
    )
    if raw_cells:
        write_csv(
            cells_to_frame(raw_cells),
            out_dir / "tb_live_metrics.csv",
            sort_by=["scheme", "source_slice", "target", "model"],
        )
    k1_path = write_tb_only_report(
        out_dir / "tb_live_eval.md",
        cells,
        summary=(
            "K1 — Logistic G4 (ledger_basic) and G2 (time_only) on `tb_live` "
            "alone under LORO. Run-level bootstrap CIs (B=1000)."
        ),
    )

    # K3 — qualitative rollup using G4 predictions for y_success_eventual
    sub_tb = checkpoints_df[checkpoints_df["source"] == TB_LIVE]
    qualitative_path = out_dir / "tb_live_qualitative.md"
    if sub_tb["run_id"].nunique() >= 2:
        split = loro(sub_tb)
        preds = predict_cell(
            checkpoints_df=sub_tb,
            labels_df=labels_df,
            target="y_success_eventual",
            spec=LEDGER_BASIC,
            split=split,
            sources_in_train=(TB_LIVE,),
        )
    else:
        preds = pd.DataFrame()
    shapes_df_all = shape_rows_for_source(TB_LIVE)
    shapes_df = pd.DataFrame(shapes_df_all) if shapes_df_all else None
    rollups = build_rollups(
        checkpoints_df=checkpoints_df,
        predictions_df=preds,
        shapes_df=shapes_df,
        final_success=_final_success_map(labels_df),
    )
    write_qualitative_report(
        qualitative_path,
        rollups,
        summary=(
            "K3 — one rollup across the TB-12 cohort. Stuck-loop precursor "
            "checkpoint = first step where `no_progress_window_5 >= 5`. "
            "Max Δp uses ledger_basic on `y_success_eventual` predictions."
        ),
    )
    return k1_path


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
    print(f"wrote tb_live K1+K3 to {out.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
