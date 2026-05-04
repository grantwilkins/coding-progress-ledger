"""Instability features: counts of disruption events and progress drops.

A "drop" is a strict decrease in the coding-categories progress series
between consecutive ledger steps in the prefix. Both reopens and
invalidations of completed leaves can cause drops; so can
add_subtask/split events that introduce new active leaves and dilute
progress (denominator growth).
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import EventType, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score

from coding_estimator.checkpoints.replay import ReplayState
from coding_estimator.ingest.run_record import RunRecord

GROUP = "instability"
COLUMNS: tuple[str, ...] = (
    "num_reopens_so_far",
    "num_invalidations_so_far",
    "num_deletes_so_far",
    "largest_progress_drop_so_far",
    "num_progress_drops_so_far",
    "steps_since_last_drop",
)


def _progress_series(events: tuple, t: int) -> list[float]:
    """Compute coding_progress at every step from min step to t. We
    re-replay at each step prefix to inherit upstream's exact semantics
    rather than re-implementing them."""
    if not events:
        return []
    s_min = min(e.step for e in events)
    series = []
    for step in range(s_min, t + 1):
        prefix = [e for e in events if e.step <= step]
        if not prefix:
            series.append(0.0)
            continue
        ledger = replay(prefix)
        series.append(score(ledger, categories=CODING_CATEGORIES).progress)
    return series


def compute(state: ReplayState, run: RunRecord) -> dict[str, Any]:
    events = state.events_so_far
    t = state.t_step

    reopens = sum(1 for e in events if e.event_type is EventType.REOPEN_SUBTASK)
    invalidations = sum(1 for e in events if e.event_type is EventType.INVALIDATE_SUBTASK)
    deletes = sum(1 for e in events if e.event_type is EventType.DELETE_SUBTASK)

    series = _progress_series(events, t)
    drops_count = 0
    largest = 0.0
    last_drop_step: int | None = None
    s_min = min((e.step for e in events), default=0)
    prev = 0.0
    for offset, value in enumerate(series):
        step = s_min + offset
        delta = max(0.0, prev - value)
        if delta > 1e-9:
            drops_count += 1
            largest = max(largest, delta)
            last_drop_step = step
        prev = value

    steps_since_drop = (t - last_drop_step) if last_drop_step is not None else None

    # `run` is currently unused but its presence in the API signals that
    # builders may need broader run context in future groups.
    del run

    return {
        "num_reopens_so_far": reopens,
        "num_invalidations_so_far": invalidations,
        "num_deletes_so_far": deletes,
        "largest_progress_drop_so_far": largest,
        "num_progress_drops_so_far": drops_count,
        "steps_since_last_drop": steps_since_drop,
    }
