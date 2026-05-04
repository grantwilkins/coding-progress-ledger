"""Stalling features: how blocked/idle the run is at the checkpoint.

`blocked_*_count` come from upstream Subtask.status. The `steps_since_*`
features measure the gap between t and the most recent occurrence of an
event class. `no_progress_window_{5,10}` are bool flags fired when
coding_progress has not increased over the trailing window.

`repeated_observation_loop_flag` is reserved upstream for transcript-
level observation events; the ledger does not carry them, so this flag
is always False on v0 ledgers (semantic:
APPLICABLE_NEVER_OBSERVED_IN_RUN, fill 0/False).
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import EventType, Status, SubtaskCategory
from ledger_progress.queries import CODING_CATEGORIES

from coding_estimator.checkpoints.features.instability import _progress_series
from coding_estimator.checkpoints.replay import ReplayState

GROUP = "stalling"
COLUMNS: tuple[str, ...] = (
    "blocked_leaf_count",
    "blocked_coding_leaf_count",
    "blocked_validation_leaf_count",
    "steps_since_completion",
    "steps_since_progress_increase",
    "steps_since_status_change",
    "steps_since_evidence",
    "repeated_observation_loop_flag",
    "no_progress_window_5",
    "no_progress_window_10",
)


def _count_blocked(state: ReplayState, categories: tuple | None) -> int:
    return sum(
        1
        for s in state.ledger.subtasks.values()
        if s.status is Status.BLOCKED
        and (categories is None or s.category in categories)
    )


def _no_progress_window(series: list[float], window: int) -> bool:
    """True iff progress has been non-increasing over the last `window`
    consecutive steps (and we have enough history to evaluate)."""
    if len(series) <= window:
        return False
    tail = series[-(window + 1):]
    return all(tail[i + 1] <= tail[i] + 1e-12 for i in range(len(tail) - 1))


def compute(state: ReplayState) -> dict[str, Any]:
    events = state.events_so_far
    t = state.t_step

    overall_blocked = _count_blocked(state, None)
    coding_blocked = _count_blocked(state, CODING_CATEGORIES)
    validation_blocked = _count_blocked(state, (SubtaskCategory.VALIDATION,))

    last_complete_step: int | None = None
    last_status_change_step: int | None = None
    last_evidence_step: int | None = None
    for e in events:
        if e.event_type is EventType.UPDATE_STATUS:
            last_status_change_step = e.step
            if e.payload.get("status") == "complete":
                last_complete_step = e.step
            if e.payload.get("evidence"):
                last_evidence_step = e.step
        elif e.event_type is EventType.ADD_EVIDENCE:
            last_evidence_step = e.step

    series = _progress_series(events, t)
    last_increase_step: int | None = None
    s_min = min((e.step for e in events), default=0)
    for offset in range(1, len(series)):
        if series[offset] > series[offset - 1] + 1e-12:
            last_increase_step = s_min + offset

    def _since(step: int | None) -> int | None:
        return (t - step) if step is not None else None

    return {
        "blocked_leaf_count": overall_blocked,
        "blocked_coding_leaf_count": coding_blocked,
        "blocked_validation_leaf_count": validation_blocked,
        "steps_since_completion": _since(last_complete_step),
        "steps_since_progress_increase": _since(last_increase_step),
        "steps_since_status_change": _since(last_status_change_step),
        "steps_since_evidence": _since(last_evidence_step),
        "repeated_observation_loop_flag": False,
        "no_progress_window_5": _no_progress_window(series, 5),
        "no_progress_window_10": _no_progress_window(series, 10),
    }


__all__ = ["GROUP", "COLUMNS", "compute"]
