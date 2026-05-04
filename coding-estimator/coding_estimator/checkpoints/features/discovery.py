"""Discovery features: cumulative counts of new-work events through t.

`num_adds_so_far`        — count of ADD_SUBTASK events in the prefix.
`num_splits_so_far`      — count of SPLIT_SUBTASK events.
`denominator_growth_so_far` — number of new active leaves added by both
                                add and split events. Each split that
                                produces N children adds N to the
                                denominator (the parent stops being a
                                leaf because it now has active children).
`steps_since_new_subtask` — number of steps since the most recent add
                            or split event. None if no such event yet.
`new_leaf_count_last_{1,3,5}_steps` — fresh leaves added in the last
                            k steps (look-back window).
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import EventType, LedgerEvent

from coding_estimator.checkpoints.replay import ReplayState

GROUP = "discovery"
COLUMNS: tuple[str, ...] = (
    "num_adds_so_far",
    "num_splits_so_far",
    "denominator_growth_so_far",
    "steps_since_new_subtask",
    "new_leaf_count_last_1_steps",
    "new_leaf_count_last_3_steps",
    "new_leaf_count_last_5_steps",
)


def _new_leaves_added(event: LedgerEvent) -> int:
    if event.event_type is EventType.ADD_SUBTASK:
        return 1
    if event.event_type is EventType.SPLIT_SUBTASK:
        children = event.payload.get("children") or []
        return len(children)
    return 0


def compute(state: ReplayState) -> dict[str, Any]:
    events = state.events_so_far
    t = state.t_step

    adds = sum(1 for e in events if e.event_type is EventType.ADD_SUBTASK)
    splits = sum(1 for e in events if e.event_type is EventType.SPLIT_SUBTASK)
    denominator_growth = sum(_new_leaves_added(e) for e in events)

    last_new_step = max(
        (
            e.step
            for e in events
            if e.event_type in {EventType.ADD_SUBTASK, EventType.SPLIT_SUBTASK}
        ),
        default=None,
    )
    steps_since_new = (t - last_new_step) if last_new_step is not None else None

    def _window(k: int) -> int:
        cutoff = t - k
        return sum(_new_leaves_added(e) for e in events if e.step > cutoff)

    return {
        "num_adds_so_far": adds,
        "num_splits_so_far": splits,
        "denominator_growth_so_far": denominator_growth,
        "steps_since_new_subtask": steps_since_new,
        "new_leaf_count_last_1_steps": _window(1),
        "new_leaf_count_last_3_steps": _window(3),
        "new_leaf_count_last_5_steps": _window(5),
    }
