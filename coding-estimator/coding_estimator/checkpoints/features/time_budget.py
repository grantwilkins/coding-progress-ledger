"""Time-budget features: step counts and (when wallclock available)
elapsed wall-clock seconds.

`elapsed_steps` is always populated. Wallclock features are populated
only on sources where `populated_on` includes them; for sources that
DON'T expose wallclock, the registry's `canonical_fill_for(source)`
returns None per the four-valued missingness contract.

`fraction_timeout_consumed` and `remaining_timeout_budget` are
tb_live-only and require a per-task timeout that the run side does
not currently expose. Until D4 wires that in, these features return
None on every run (semantic: UNKNOWN_DUE_TO_MISSING_ARTIFACT).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ledger_progress.core import EventType

from coding_estimator.checkpoints.replay import ReplayState
from coding_estimator.ingest.run_record import RunRecord

GROUP = "time_budget"
COLUMNS: tuple[str, ...] = (
    "elapsed_steps",
    "elapsed_wall_time",
    "fraction_timeout_consumed",
    "remaining_timeout_budget",
    "completion_rate_recent_steps",
)

_RECENT_WINDOW = 5


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(text)


def compute(state: ReplayState, run: RunRecord) -> dict[str, Any]:
    events = state.events_so_far
    t = state.t_step
    s_min = min((e.step for e in events), default=0)
    elapsed_steps = t - s_min

    # Wallclock: populated iff the run has real wallclock AND we have
    # both a first and a current event with a timestamp.
    elapsed_wall_time: float | None = None
    if run.has_real_wallclock and events:
        first_ts = _parse_iso(events[0].timestamp)
        last_ts = _parse_iso(events[-1].timestamp)
        if first_ts and last_ts:
            elapsed_wall_time = (last_ts - first_ts).total_seconds()

    # tb_live timeout is not exposed by the run-side artifact yet.
    fraction_timeout = None
    remaining_timeout = None

    completions_in_window = sum(
        1
        for e in events
        if e.event_type is EventType.UPDATE_STATUS
        and e.payload.get("status") == "complete"
        and e.step > t - _RECENT_WINDOW
    )
    completion_rate = completions_in_window / _RECENT_WINDOW

    return {
        "elapsed_steps": elapsed_steps,
        "elapsed_wall_time": elapsed_wall_time,
        "fraction_timeout_consumed": fraction_timeout,
        "remaining_timeout_budget": remaining_timeout,
        "completion_rate_recent_steps": completion_rate,
    }
