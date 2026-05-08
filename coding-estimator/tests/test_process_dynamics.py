"""
Claim:
- The process-dynamics audit helpers rebuild future-drop witnesses using
  a strict future window, preserve lead-time semantics, and mask
  near-finish checkpoints by omission.
- The audit-only target variants catch near-boundary duplication rather
  than rewriting the underlying label table.
- Validation-run auditing distinguishes between validation followed by
  discovery within h5, after h5, and not at all.

Plausible wrong implementations:
- Count the drop event at step t as "future", creating same-step
  leakage.
- Keep every positive checkpoint for the same realized drop episode,
  which would inflate the easiest near-boundary cases.
- Treat `lead_ge_2` as a row filter instead of a label rewrite.
- Count discovery with no prior validation transition as a
  validation-new-work signal.
"""

from __future__ import annotations

from pathlib import Path

from ledger_progress.serialization import event_from_dict

from coding_estimator.eval.process_dynamics import (
    _validation_run_stats,
    apply_progress_drop_variant,
    build_progress_drop_witness_rows,
)
from coding_estimator.ingest.run_record import RunRecord


def _run_from_dicts(events: list[dict], *, run_id: str = "synth") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        source="tb_live_v2",
        ledger_path=Path("/tmp/synth-ledger.jsonl"),
        events=tuple(event_from_dict(event) for event in events),
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id=run_id,
        task_family="progress_drop",
        arm="A",
        difficulty="medium",
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


def _two_drop_run() -> RunRecord:
    return _run_from_dicts(
        [
            {
                "step": 0,
                "event_type": "init",
                "subtask_id": None,
                "payload": {"root_task": "synth"},
                "reason": None,
                "timestamp": "2026-05-05T00:00:00Z",
            },
            {
                "step": 1,
                "event_type": "add_subtask",
                "subtask_id": "p1",
                "payload": {"description": "p1", "category": "product", "weight": 1.0, "parent_id": None},
                "reason": None,
                "timestamp": "2026-05-05T00:00:01Z",
            },
            {
                "step": 2,
                "event_type": "update_status",
                "subtask_id": "p1",
                "payload": {"status": "complete", "evidence": ["done"]},
                "reason": None,
                "timestamp": "2026-05-05T00:00:02Z",
            },
            {
                "step": 4,
                "event_type": "add_subtask",
                "subtask_id": "p2",
                "payload": {"description": "p2", "category": "product", "weight": 1.0, "parent_id": None},
                "reason": None,
                "timestamp": "2026-05-05T00:00:04Z",
            },
            {
                "step": 5,
                "event_type": "update_status",
                "subtask_id": "p2",
                "payload": {"status": "complete", "evidence": ["done"]},
                "reason": None,
                "timestamp": "2026-05-05T00:00:05Z",
            },
            {
                "step": 7,
                "event_type": "add_subtask",
                "subtask_id": "p3",
                "payload": {"description": "p3", "category": "product", "weight": 1.0, "parent_id": None},
                "reason": None,
                "timestamp": "2026-05-05T00:00:07Z",
            },
            {
                "step": 8,
                "event_type": "update_status",
                "subtask_id": "p3",
                "payload": {"status": "complete", "evidence": ["done"]},
                "reason": None,
                "timestamp": "2026-05-05T00:00:08Z",
            },
        ]
    )


def _validation_run(*, discovery_step: int | None, include_validation: bool = True) -> RunRecord:
    events = [
        {
            "step": 0,
            "event_type": "init",
            "subtask_id": None,
            "payload": {"root_task": "validation"},
            "reason": None,
            "timestamp": "2026-05-05T00:00:00Z",
        },
        {
            "step": 1,
            "event_type": "add_subtask",
            "subtask_id": "v1",
            "payload": {"description": "validate", "category": "validation", "weight": 1.0, "parent_id": None},
            "reason": None,
            "timestamp": "2026-05-05T00:00:01Z",
        },
    ]
    if include_validation:
        events.append(
            {
                "step": 3,
                "event_type": "update_status",
                "subtask_id": "v1",
                "payload": {"status": "complete", "evidence": ["validated"]},
                "reason": None,
                "timestamp": "2026-05-05T00:00:03Z",
            }
        )
    if discovery_step is not None:
        events.append(
            {
                "step": discovery_step,
                "event_type": "add_subtask",
                "subtask_id": "p1",
                "payload": {"description": "new work", "category": "product", "weight": 1.0, "parent_id": None},
                "reason": None,
                "timestamp": f"2026-05-05T00:00:{discovery_step:02d}Z",
            }
        )
    return _run_from_dicts(events, run_id=f"validation_{discovery_step}_{include_validation}")


def test_progress_drop_witness_uses_strict_future_window_and_no_same_step_leakage() -> None:
    rows = build_progress_drop_witness_rows(_two_drop_run(), checkpoint_steps=list(range(9)), horizon=3)
    by_step = {row["checkpoint_step"]: row for row in rows}
    assert by_step[2]["label"] == 1
    assert by_step[2]["next_drop_step"] == 4
    assert by_step[2]["lead_time"] == 2
    assert by_step[2]["drop_magnitude"] == 0.5
    assert by_step[3]["label"] == 1
    assert by_step[3]["lead_time"] == 1
    assert by_step[4]["label"] == 0
    assert by_step[4]["next_drop_step"] is None


def test_progress_drop_witness_masks_near_finish_by_omitting_rows() -> None:
    rows = build_progress_drop_witness_rows(_two_drop_run(), checkpoint_steps=list(range(9)), horizon=3)
    seen_steps = {row["checkpoint_step"] for row in rows}
    assert 6 not in seen_steps
    assert 7 not in seen_steps
    assert 8 not in seen_steps


def test_progress_drop_variants_enforce_lead_and_first_positive_per_episode() -> None:
    witness = build_progress_drop_witness_rows(_two_drop_run(), checkpoint_steps=list(range(9)), horizon=3)
    base = [
        {
            **row,
            "checkpoint_id": f"synth::{row['checkpoint_step']}",
            "task_id": "synth",
            "task_family": "progress_drop",
            "arm": "A",
            "model_name": "m",
            "coding_progress": row["current_progress"],
        }
        for row in witness
    ]
    import pandas as pd

    df = pd.DataFrame(base)
    lead = apply_progress_drop_variant(df, "h5_first_drop_lead_ge_2")
    by_step = {int(row["checkpoint_step"]): int(row["label"]) for _, row in lead.iterrows()}
    assert by_step[2] == 1
    assert by_step[3] == 0

    first_only = apply_progress_drop_variant(df, "h5_first_positive_per_drop_episode")
    positive_steps = first_only.loc[first_only["label"] > 0, "checkpoint_step"].astype(int).tolist()
    assert positive_steps == [2, 5]


def test_progress_drop_variant_excludes_add_split_reopen_checkpoint_steps() -> None:
    witness = build_progress_drop_witness_rows(_two_drop_run(), checkpoint_steps=list(range(9)), horizon=3)
    import pandas as pd

    df = pd.DataFrame(
        {
            **row,
            "checkpoint_id": f"synth::{row['checkpoint_step']}",
            "task_id": "synth",
            "task_family": "progress_drop",
            "arm": "A",
            "model_name": "m",
            "coding_progress": row["current_progress"],
        }
        for row in witness
    )
    filtered = apply_progress_drop_variant(
        df,
        "h5_first_drop_lead_ge_2_excluding_checkpoint_steps_with_add_split_reopen",
    )
    assert 2 in filtered["checkpoint_step"].tolist()
    assert 4 not in filtered["checkpoint_step"].tolist()


def test_validation_stats_distinguish_within_h5_after_h5_and_missing_validation() -> None:
    within = _validation_run_stats(_validation_run(discovery_step=5, include_validation=True))
    after = _validation_run_stats(_validation_run(discovery_step=9, include_validation=True))
    none = _validation_run_stats(_validation_run(discovery_step=None, include_validation=True))
    no_validation = _validation_run_stats(_validation_run(discovery_step=5, include_validation=False))

    assert within["has_validation_transition"] is True
    assert within["has_discovery_or_reopen_after_validation_within_5"] is True
    assert within["has_discovery_or_reopen_after_validation_any_later"] is True

    assert after["has_validation_transition"] is True
    assert after["has_discovery_or_reopen_after_validation_within_5"] is False
    assert after["has_discovery_or_reopen_after_validation_any_later"] is True

    assert none["has_validation_transition"] is True
    assert none["has_discovery_or_reopen_after_validation_within_5"] is False
    assert none["has_discovery_or_reopen_after_validation_any_later"] is False

    assert no_validation["has_validation_transition"] is False
    assert no_validation["has_discovery_or_reopen_after_validation_within_5"] is False
    assert no_validation["has_discovery_or_reopen_after_validation_any_later"] is False

