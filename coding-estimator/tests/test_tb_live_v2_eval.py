"""
Claim:
- `exact_task_group_map()` and `build_tb_live_v2_splits()` use exact
  `task_id` groups for `tb_live_v2`, so same-task multi-arm replicas
  stay in one LTFO fold even when every run shares the same coarse
  `task_family`.
- `render_tb_live_v2_report()` makes exact-task-holdout process
  dynamics the headline and keeps overlap-heavy splits auxiliary only.
- Terminal-success reporting is explicitly caveated for arm
  concentration, ceiling effects, and the `solution.sh` optimism note.

Plausible wrong implementations:
- Group LTFO by coarse `task_family`, collapsing all
  `validation_new_work_*` tasks into one fold and making exact-task
  holdout impossible.
- Put LORO / random holdout rows into the headline table because they
  have bigger N and look better.
- Surface terminal success as an unqualified headline despite the task
  acceptance text requiring arm/ceiling/solution caveats.
"""

from __future__ import annotations

import pandas as pd

from coding_estimator.eval.harness import EvalCell
from coding_estimator.eval.tb_live_v2 import (
    TBLiveV2Profile,
    build_tb_live_v2_splits,
    exact_task_group_map,
    render_tb_live_v2_report,
)


def _checkpoint_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for run_id, task_id in (
        ("task_a__armA__001", "task_a"),
        ("task_a__armB__002", "task_a"),
        ("task_b__armA__003", "task_b"),
        ("task_b__armB__004", "task_b"),
    ):
        for step in range(3):
            rows.append(
                {
                    "run_id": run_id,
                    "source": "tb_live_v2",
                    "checkpoint_id": f"{run_id}::{step}",
                    "checkpoint_step": step,
                    "task_id": task_id,
                    "task_family": "validation_new_work",
                    "elapsed_steps": float(step),
                    "coding_progress": float(step) / 2.0,
                    "validation_started": bool(step >= 1),
                    "validation_complete": bool(step >= 2),
                    "blocked_leaf_count": 0.0,
                    "active_leaf_count": 1.0,
                    "completed_leaf_count": float(step),
                    "validation_progress": float(step) / 2.0,
                    "product_progress": float(step) / 2.0,
                    "investigation_progress": 0.0,
                    "num_adds_so_far": float(step),
                    "num_splits_so_far": 0.0,
                    "denominator_growth_so_far": 0.0,
                    "steps_since_new_subtask": float(step),
                    "num_reopens_so_far": 0.0,
                    "num_invalidations_so_far": 0.0,
                    "num_deletes_so_far": 0.0,
                    "largest_progress_drop_so_far": 0.0,
                    "num_progress_drops_so_far": 0.0,
                    "steps_since_last_drop": float(step),
                    "active_coding_leaf_count": 1.0,
                    "active_validation_leaf_count": 0.0,
                    "blocked_coding_leaf_count": 0.0,
                    "blocked_validation_leaf_count": 0.0,
                    "steps_since_completion": float(step),
                    "steps_since_progress_increase": 0.0,
                    "steps_since_status_change": 0.0,
                    "steps_since_evidence": float(step),
                    "repeated_observation_loop_flag": False,
                    "no_progress_window_5": False,
                    "no_progress_window_10": False,
                    "validation_leaf_exists": True,
                    "validation_failed": False,
                    "validation_blocked": False,
                    "validation_in_progress": False,
                    "num_validation_attempts": float(step),
                    "num_validation_failures": 0.0,
                    "num_validation_successes": float(step >= 2),
                    "steps_since_last_validation": float(step),
                    "submit_without_validation_so_far": False,
                    "strong_completion_count": float(step),
                    "manual_only_completion_count": 0.0,
                    "weak_product_completion_count": 0.0,
                    "strong_evidence_fraction": 0.5,
                    "manual_only_evidence_fraction": 0.0,
                    "elapsed_wall_time": float(step),
                    "fraction_timeout_consumed": None,
                    "remaining_timeout_budget": None,
                    "completion_rate_recent_steps": 0.5,
                }
            )
    return pd.DataFrame(rows)


def _labels_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for run_id, task_id, success in (
        ("task_a__armA__001", "task_a", 1.0),
        ("task_a__armB__002", "task_a", 0.0),
        ("task_b__armA__003", "task_b", 1.0),
        ("task_b__armB__004", "task_b", 0.0),
    ):
        for step in range(3):
            cid = f"{run_id}::{step}"
            rows.extend(
                [
                    {
                        "run_id": run_id,
                        "source": "tb_live_v2",
                        "checkpoint_id": cid,
                        "target_name": "y_success_eventual",
                        "label_value": success,
                        "is_masked": False,
                        "task_id": task_id,
                    },
                    {
                        "run_id": run_id,
                        "source": "tb_live_v2",
                        "checkpoint_id": cid,
                        "target_name": "y_future_progress_drop_h5",
                        "label_value": float((run_id.endswith("002") or run_id.endswith("004")) and step >= 1),
                        "is_masked": False,
                        "task_id": task_id,
                    },
                    {
                        "run_id": run_id,
                        "source": "tb_live_v2",
                        "checkpoint_id": cid,
                        "target_name": "y_validation_new_work_h5",
                        "label_value": float((run_id.endswith("001") or run_id.endswith("003")) and step == 1),
                        "is_masked": False,
                        "task_id": task_id,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "task_a__armA__001",
                "source": "tb_live_v2",
                "task_id": "task_a",
                "task_family": "validation_new_work",
                "arm": "A",
                "model_name": "opus",
                "final_success": True,
            },
            {
                "run_id": "task_a__armB__002",
                "source": "tb_live_v2",
                "task_id": "task_a",
                "task_family": "validation_new_work",
                "arm": "B",
                "model_name": "sonnet",
                "final_success": False,
            },
            {
                "run_id": "task_b__armA__003",
                "source": "tb_live_v2",
                "task_id": "task_b",
                "task_family": "validation_new_work",
                "arm": "A",
                "model_name": "opus",
                "final_success": True,
            },
            {
                "run_id": "task_b__armB__004",
                "source": "tb_live_v2",
                "task_id": "task_b",
                "task_family": "validation_new_work",
                "arm": "B",
                "model_name": "sonnet",
                "final_success": False,
            },
        ]
    )


def _cell(
    *,
    scheme: str,
    source_slice: str,
    target: str,
    model: str,
) -> EvalCell:
    return EvalCell(
        target=target,
        model=model,
        scheme=scheme,
        source_slice=source_slice,
        feasible=True,
        n_runs_train=10,
        n_runs_test=4,
        n_checkpoints_test=20,
        positive_rate_data=0.4,
        predicted_positive_rate=0.35,
        auroc=0.7,
        brier=0.2,
        log_loss=0.5,
        ece=0.1,
        brier_ci_low=0.15,
        brier_ci_high=0.25,
        note=None,
    )


def test_exact_task_group_map_prefers_task_id_over_coarse_family() -> None:
    groups = exact_task_group_map(
        checkpoints_df=_checkpoint_frame(),
        labels_df=_labels_frame(),
        manifest_df=_manifest_frame(),
    )
    assert groups["task_a__armA__001"] == "task_a"
    assert groups["task_a__armB__002"] == "task_a"
    assert groups["task_b__armA__003"] == "task_b"
    assert groups["task_b__armB__004"] == "task_b"
    assert len(set(groups.values())) == 2


def test_build_tb_live_v2_splits_keeps_same_task_multiarm_runs_in_one_ltfo_fold() -> None:
    splits = build_tb_live_v2_splits(
        checkpoints_df=_checkpoint_frame(),
        labels_df=_labels_frame(),
        manifest_df=_manifest_frame(),
    )
    ltfo = splits["ltfo"]
    by_fold = {f.fold_id: set(f.test_run_ids) for f in ltfo.folds}
    assert by_fold["ltfo::task_a"] == {"task_a__armA__001", "task_a__armB__002"}
    assert by_fold["ltfo::task_b"] == {"task_b__armA__003", "task_b__armB__004"}


def test_render_tb_live_v2_report_headline_is_exact_task_process_dynamics_only() -> None:
    profile = TBLiveV2Profile(
        n_runs=102,
        n_success=81,
        n_fail=21,
        n_unresolved=0,
        n_exact_tasks=25,
        n_coarse_families=5,
        exact_task_group_sizes=((3, 16), (6, 9)),
        arm_rows=(),
        family_rows=(),
        failure_task_rows=(),
    )
    cells = [
        _cell(
            scheme="ltfo",
            source_slice="exact-task-holdout",
            target="y_future_progress_drop_h5",
            model="ledger_basic",
        ),
        _cell(
            scheme="ltfo",
            source_slice="exact-task-holdout",
            target="y_validation_new_work_h5",
            model="time_only",
        ),
        _cell(
            scheme="loro",
            source_slice="loro-overlap",
            target="y_future_progress_drop_h5",
            model="ledger_basic",
        ),
        _cell(
            scheme="holdout",
            source_slice="holdout-overlap",
            target="y_success_eventual",
            model="time_only",
        ),
    ]
    report = render_tb_live_v2_report(cells=cells, profile=profile)
    headline_block = report.split("## Auxiliary only: overlap-heavy splits", 1)[0]
    assert "exact-task-holdout" in headline_block
    assert "y_future_progress_drop_h5" in headline_block
    assert "y_validation_new_work_h5" in headline_block
    assert "loro-overlap" not in headline_block
    assert "holdout-overlap" not in headline_block
    assert "y_success_eventual" not in headline_block


def test_render_tb_live_v2_report_caveats_terminal_success() -> None:
    profile = TBLiveV2Profile(
        n_runs=102,
        n_success=81,
        n_fail=21,
        n_unresolved=0,
        n_exact_tasks=25,
        n_coarse_families=5,
        exact_task_group_sizes=((3, 16), (6, 9)),
        arm_rows=(
            type("Row", (), {"name": "A", "n_success": 33, "n_runs": 34})(),
            type("Row", (), {"name": "B", "n_success": 24, "n_runs": 34})(),
            type("Row", (), {"name": "C", "n_success": 24, "n_runs": 34})(),
        ),
        family_rows=(),
        failure_task_rows=(),
    )
    report = render_tb_live_v2_report(
        cells=[_cell(
            scheme="ltfo",
            source_slice="exact-task-holdout",
            target="y_success_eventual",
            model="ledger_basic",
        )],
        profile=profile,
    )
    assert "Arm concentration is strong" in report
    assert "ceiling-limited" in report
    assert "`solution.sh` was present" in report
