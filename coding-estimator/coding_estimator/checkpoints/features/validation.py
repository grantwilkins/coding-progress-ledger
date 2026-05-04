"""Validation features: status of validation subtasks at the checkpoint.

`submit_without_validation_so_far` is the prefix-only analogue of the
terminal `final_artifact_without_validation`. It is True iff *no
validation events* have been observed by step t. The terminal column
is forbidden as a feature (see schemas/forbidden_columns.json); this
prefix-only flag is allowed.
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import EventType, Status, SubtaskCategory

from coding_estimator.checkpoints.replay import ReplayState

GROUP = "validation"
COLUMNS: tuple[str, ...] = (
    "validation_leaf_exists",
    "validation_started",
    "validation_complete",
    "validation_failed",
    "validation_blocked",
    "validation_in_progress",
    "num_validation_attempts",
    "num_validation_failures",
    "num_validation_successes",
    "steps_since_last_validation",
    "submit_without_validation_so_far",
)


def _validation_subtask_ids(state: ReplayState) -> set[str]:
    return {
        sid
        for sid, s in state.ledger.subtasks.items()
        if s.category is SubtaskCategory.VALIDATION
    }


def compute(state: ReplayState) -> dict[str, Any]:
    val_ids = _validation_subtask_ids(state)
    val_subtasks = [state.ledger.subtasks[sid] for sid in val_ids]
    has_any = bool(val_subtasks)
    started = any(
        s.status in {Status.IN_PROGRESS, Status.COMPLETE, Status.BLOCKED, Status.INVALIDATED}
        for s in val_subtasks
    )
    has_complete = any(s.status is Status.COMPLETE for s in val_subtasks)
    has_failed = any(s.status is Status.INVALIDATED for s in val_subtasks)
    has_blocked = any(s.status is Status.BLOCKED for s in val_subtasks)
    has_in_progress = any(s.status is Status.IN_PROGRESS for s in val_subtasks)

    val_events = [e for e in state.events_so_far if e.subtask_id in val_ids]
    attempts = sum(
        1 for e in val_events if e.event_type is EventType.UPDATE_STATUS
    )
    failures = sum(
        1 for e in val_events if e.event_type is EventType.INVALIDATE_SUBTASK
    )
    successes = sum(
        1
        for e in val_events
        if e.event_type is EventType.UPDATE_STATUS
        and e.payload.get("status") == "complete"
    )

    last_val_step = max((e.step for e in val_events), default=None)
    steps_since = (state.t_step - last_val_step) if last_val_step is not None else None

    return {
        "validation_leaf_exists": has_any,
        "validation_started": started,
        "validation_complete": has_complete,
        "validation_failed": has_failed,
        "validation_blocked": has_blocked,
        "validation_in_progress": has_in_progress,
        "num_validation_attempts": attempts,
        "num_validation_failures": failures,
        "num_validation_successes": successes,
        "steps_since_last_validation": steps_since,
        # Prefix-only: True iff NO validation event has been observed yet.
        "submit_without_validation_so_far": not has_any,
    }
