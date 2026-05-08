"""
Claim:
- The frozen tb_live_v2 artifacts support a reproducible process-dynamics
  audit surface: witness rows reproduce the shipped unmasked drop labels,
  exact-task OOF predictions cover the same 213 rows, validation-new-work
  remains zero-positive on this substrate, and ranking-based case
  selection is deterministic.

Plausible wrong implementations:
- Witness reconstruction drifts from the shipped label contract.
- OOF export accidentally uses a different row set than the frozen
- exact-task evaluation.
- Validation auditing counts positives that do not exist in the current
  frozen corpus.
- Case-study selection depends on unstable row ordering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from coding_estimator.checkpoints.fills import apply_canonical_fills
from coding_estimator.eval.process_dynamics import (
    build_exact_task_oof_predictions,
    build_progress_drop_witness,
    build_validation_new_work_audit,
    select_case_studies,
    verify_progress_drop_witness,
)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    checkpoints = apply_canonical_fills(pd.read_parquet(root / "datasets/checkpoints_tb_live_v2.parquet"))
    labels = pd.read_parquet(root / "datasets/labels_tb_live_v2.parquet")
    manifest = pd.read_csv(root / "datasets/manifests/tb_live_v2.csv")
    return checkpoints, labels, manifest


def test_tb_live_v2_witness_matches_shipped_drop_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoints, labels, _ = _load_inputs()
    witness = build_progress_drop_witness(
        checkpoints_df=checkpoints,
        labels_df=labels,
        runs_dir=root / "runs/tb_live_v2",
        horizon=5,
    )
    _, mismatches = verify_progress_drop_witness(witness)
    assert len(witness) == 213
    assert mismatches.empty


def test_tb_live_v2_exact_task_oof_export_has_expected_row_count() -> None:
    checkpoints, labels, manifest = _load_inputs()
    oof, _ = build_exact_task_oof_predictions(
        checkpoints_df=checkpoints,
        labels_df=labels,
        manifest_df=manifest,
    )
    assert len(oof) == 213


def test_tb_live_v2_validation_audit_reproduces_zero_positive_status() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoints, labels, manifest = _load_inputs()
    audit_df, recommendation = build_validation_new_work_audit(
        checkpoints_df=checkpoints,
        labels_df=labels,
        manifest_df=manifest,
        runs_dir=root / "runs/tb_live_v2",
    )
    all_slice = audit_df.set_index("slice_name").loc["all_tb_live_v2_runs"]
    assert int(all_slice["n_unmasked_positive_checkpoints_current_label"]) == 0
    assert recommendation == "defer_on_tb_live_v2"


def test_tb_live_v2_case_selection_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoints, labels, manifest = _load_inputs()
    witness = build_progress_drop_witness(
        checkpoints_df=checkpoints,
        labels_df=labels,
        runs_dir=root / "runs/tb_live_v2",
        horizon=5,
    )
    oof, _ = build_exact_task_oof_predictions(
        checkpoints_df=checkpoints,
        labels_df=labels,
        manifest_df=manifest,
    )
    first = select_case_studies(
        oof_df=oof,
        witness_df=witness,
        figures_dir=root / "reports/figures",
    )
    second = select_case_studies(
        oof_df=oof,
        witness_df=witness,
        figures_dir=root / "reports/figures",
    )
    first_keys = [(case.section_title, case.run_id, case.checkpoint_step) for case in first]
    second_keys = [(case.section_title, case.run_id, case.checkpoint_step) for case in second]
    assert first_keys == second_keys

