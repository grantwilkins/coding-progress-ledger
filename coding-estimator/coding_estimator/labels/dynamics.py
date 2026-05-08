"""Process-dynamics labels (h5).

  y_future_progress_drop_h5    — strict-future progress drop in (t, t+5]
  y_validation_new_work_h5     — validation transition in (t, t+5] followed
                                  by a discovery (PRODUCT/INVESTIGATION add
                                  or reopen) inside the same window.

Definitions are pinned to the upstream snapshot in
`_upstream_q_snapshot.py`. Both labels are masked when
`t + 5 > finish_step` OR the checkpoint is the run's terminal step.

The horizon is half-open `(t, t+H]` (upstream Q convention).
"""

from __future__ import annotations

from dataclasses import dataclass

from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.labels._upstream_q_snapshot import (
    _coding_progress,
    _events_through_step,
    label_future_progress_drop,
    label_validation_exposes_new_work,
)

H5 = 5


@dataclass(frozen=True)
class DynamicsLabel:
    value: bool | None
    is_masked: bool
    mask_reason: str | None


def _mask_reason(t_step: int, is_terminal: bool, finish_step: int | None) -> str | None:
    if is_terminal:
        return "is_terminal_checkpoint"
    if finish_step is None:
        return "finish_step_unknown"
    if t_step + H5 > finish_step:
        return "horizon_exceeds_finish_step"
    return None


def _coding_progress_at(events, t_step: int) -> float:
    return _coding_progress(_events_through_step(events, t_step))


def future_progress_drop_h5(
    run: RunRecord,
    t_step: int,
    *,
    is_terminal: bool,
    finish_step: int | None,
) -> DynamicsLabel:
    reason = _mask_reason(t_step, is_terminal, finish_step)
    if reason is not None:
        return DynamicsLabel(value=None, is_masked=True, mask_reason=reason)
    events = list(run.events)
    progress = _coding_progress_at(events, t_step)
    val = label_future_progress_drop(events, t_step, H5, progress)
    return DynamicsLabel(value=bool(val), is_masked=False, mask_reason=None)


def validation_new_work_h5(
    run: RunRecord,
    t_step: int,
    *,
    is_terminal: bool,
    finish_step: int | None,
) -> DynamicsLabel:
    reason = _mask_reason(t_step, is_terminal, finish_step)
    if reason is not None:
        return DynamicsLabel(value=None, is_masked=True, mask_reason=reason)
    val = label_validation_exposes_new_work(list(run.events), t_step, H5)
    return DynamicsLabel(value=bool(val), is_masked=False, mask_reason=None)
