#!/usr/bin/env python
"""P2/P3 driver — build the v0 sign-off package and READY/NOT_READY
report.

Usage:
    uv run python scripts/run_sign_off.py \\
        --checkpoints datasets/checkpoints_all.parquet \\
        --labels datasets/labels_all.parquet \\
        --estimator-id ledger_basic_v0.1 \\
        --estimator-version 0.1.0 \\
        --models-root models \\
        --reports-root reports

Builds:
    models/<id>/model_card.{json,md}
    reports/sign_off_<id>.md
    reports/{READY,NOT_READY}_FOR_SCHEDULING.md
    reports/ESTIMATOR_GO_NO_GO.md (if absent — re-uses run_go_no_go output)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from coding_estimator.baselines import LEDGER_BASIC, V0_BASELINES
from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.failure_modes import (
    HEADLINE_TARGET,
    FailureModeResult,
    evaluate_o1,
    evaluate_o5,
    evaluate_o7,
)
from coding_estimator.eval.go_no_go import evaluate_gate
from coding_estimator.eval.harness import predict_cell
from coding_estimator.eval.sign_off import (
    build_sign_off,
    write_p3_report,
    write_sign_off_summary,
)
from coding_estimator.eval.tb_qualitative import TB_LIVE
from coding_estimator.io import write_json
from coding_estimator.splits.protocol import loro

REPO_ROOT = Path(__file__).resolve().parents[1]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0000000"


def _final_success_map(labels_df: pd.DataFrame) -> dict[str, int]:
    s = labels_df[
        (labels_df["target_name"] == HEADLINE_TARGET)
        & (~labels_df["is_masked"].astype(bool))
    ].drop_duplicates("run_id")
    return {str(r["run_id"]): int(r["label_value"]) for _, r in s.iterrows()}


def _g4_per_source_predictions(
    *, checkpoints_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
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


def run(
    *,
    checkpoints_path: Path,
    labels_path: Path,
    estimator_id: str,
    estimator_version: str,
    models_root: Path,
    reports_root: Path,
    d5_audit_path: Path | None = None,
) -> dict[str, Path]:
    checkpoints_df = apply_canonical_fills(pd.read_parquet(checkpoints_path))
    labels_df = pd.read_parquet(labels_path)

    # Failure modes
    g4_preds = _g4_per_source_predictions(
        checkpoints_df=checkpoints_df, labels_df=labels_df
    )
    final_success = _final_success_map(labels_df)
    o1 = evaluate_o1(
        predictions_df=g4_preds,
        checkpoints_df=checkpoints_df,
        final_success=final_success,
    )
    o5 = evaluate_o5(
        checkpoints_df=checkpoints_df, labels_df=labels_df, test_source=TB_LIVE
    )
    o7 = evaluate_o7(checkpoints_df=checkpoints_df, labels_df=labels_df)

    # Gate
    gate = evaluate_gate(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        d5_audit_path=d5_audit_path,
    )

    bundle_dir = models_root / estimator_id
    gate_report_path = str(reports_root / "ESTIMATOR_GO_NO_GO.md")

    # Card + bundle
    feature_groups = ("closure", "frontier", "instability", "discovery")
    targets = list(_supported_targets(labels_df))
    cells = _per_source_eval_cells(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        targets=targets,
    )
    json_path, md_path, record = build_sign_off(
        estimator_id=estimator_id,
        estimator_version=estimator_version,
        model_family="ledger_basic",
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        feature_groups=feature_groups,
        targets=targets,
        eval_cells=cells,
        headline_scheme="loro",
        diagnostic_schemes=("loso",),
        headline_seed=0,
        calibration_method="raw",
        intended_use=[
            "Offline evaluation of v0 ledger-basic estimator",
            "Sign-off audit input",
        ],
        non_use_cases=[
            "Driving scheduling, modulation, or other control actions",
            "Decision-making under tight latency budgets",
        ],
        commit_sha=_commit_sha(),
        o1=o1, o5=o5, o7=o7,
        gate=gate,
        gate_report_path=gate_report_path,
        out_dir=bundle_dir,
    )

    summary_path = write_sign_off_summary(
        reports_root / f"sign_off_{estimator_id}.md",
        estimator_id=estimator_id,
        record=record,
        gate=gate,
        o1=o1, o5=o5, o7=o7,
        bundle_dir=bundle_dir,
        gate_report_path=gate_report_path,
    )
    p3_path = write_p3_report(
        reports_root,
        estimator_id=estimator_id,
        record=record,
        gate=gate,
        o1=o1, o5=o5, o7=o7,
    )
    return {
        "model_card_json": json_path,
        "model_card_md": md_path,
        "sign_off_summary": summary_path,
        "p3_report": p3_path,
    }


def _supported_targets(labels_df: pd.DataFrame) -> tuple[str, ...]:
    available = set(labels_df["target_name"].unique())
    candidates = (
        "y_success_eventual",
        "y_future_progress_drop_h5",
        "y_validation_new_work_h5",
        "y_submit_without_validation",
    )
    return tuple(t for t in candidates if t in available)


def _per_source_eval_cells(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    targets: list[str],
):
    """Build EvalCell list for the LORO scheme used in the card."""
    from coding_estimator.eval.harness import evaluate_cell

    cells = []
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        for target in targets:
            cells.append(
                evaluate_cell(
                    checkpoints_df=sub,
                    labels_df=labels_df,
                    target=target,
                    spec=LEDGER_BASIC,
                    split=split,
                    source_slice=source,
                    sources_in_train=(source,),
                    feasible=True,
                    bootstrap_b=200,
                )
            )
    return cells


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--estimator-id", type=str, default="ledger_basic_v0.1")
    p.add_argument("--estimator-version", type=str, default="0.1.0")
    p.add_argument("--models-root", type=Path, default=Path("models"))
    p.add_argument("--reports-root", type=Path, default=Path("reports"))
    p.add_argument(
        "--d5-audit",
        type=Path,
        default=Path("reports/d5_audit.json"),
        help="path to D5 audit JSON (default: reports/d5_audit.json)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = run(
        checkpoints_path=args.checkpoints,
        labels_path=args.labels,
        estimator_id=args.estimator_id,
        estimator_version=args.estimator_version,
        models_root=args.models_root,
        reports_root=args.reports_root,
        d5_audit_path=args.d5_audit,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
