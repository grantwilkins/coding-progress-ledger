"""Deadline-aware completion estimator stubs.

`p_finish_by` projects whether a ledger will reach progress=1.0 by a given
deadline using a naive linear extrapolation from observed progress velocity.
This is a documented stub, not a calibrated predictor: it assumes the
observed progress rate will continue unchanged. Use it to give downstream
code (Workstream W/Q) a callable shape, then replace the body once a real
model exists.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .core import Ledger, SubtaskCategory
from .scoring import score


CODING = (SubtaskCategory.PRODUCT, SubtaskCategory.VALIDATION, SubtaskCategory.INVESTIGATION)


def p_finish_by(
    ledger: Ledger,
    deadline: str | datetime,
    categories: Iterable[SubtaskCategory | str] | None = CODING,
) -> float:
    """Probability the ledger reaches progress=1.0 by `deadline`.

    Returns 1.0 if already at progress=1.0. Returns 0.0 if no timestamps,
    no progress yet, or the projected finish is past the deadline.
    Otherwise returns a triangular falloff between (now, projected_finish).

    Assumptions:
    - Progress accrues at the average rate observed so far.
    - Deadline is in UTC ISO 8601 if passed as a string.
    - This is a stub; not predictive without calibration.
    """
    deadline_dt = deadline if isinstance(deadline, datetime) else datetime.fromisoformat(deadline)
    timestamps = [datetime.fromisoformat(e.timestamp) for e in ledger.events if e.timestamp]
    if not timestamps:
        return 0.0

    progress = score(ledger, categories=categories).progress
    if progress >= 1.0:
        return 1.0

    first, last = timestamps[0], timestamps[-1]
    elapsed = (last - first).total_seconds()
    if elapsed <= 0 or progress <= 0:
        return 0.0

    rate_per_second = progress / elapsed
    remaining_progress = 1.0 - progress
    seconds_to_finish = remaining_progress / rate_per_second
    seconds_until_deadline = (deadline_dt - last).total_seconds()
    if seconds_until_deadline <= 0:
        return 0.0
    if seconds_until_deadline >= seconds_to_finish:
        return 1.0
    return seconds_until_deadline / seconds_to_finish
