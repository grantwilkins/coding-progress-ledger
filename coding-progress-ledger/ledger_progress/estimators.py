"""Deadline-aware completion estimator stubs.

`fraction_of_time_to_finish_remaining` projects, under a naive linear
extrapolation from observed progress velocity, what fraction of the time
needed to reach progress=1.0 is available before a given deadline. The
return is dimensionally a *time ratio*, **not a probability**: callers
that want a probability must supply their own calibration.

Returns:
- 1.0 when the projected finish fits before the deadline (or the ledger
  is already complete).
- 0.0 when there are no timestamps yet, no progress yet, or the deadline
  is in the past.
- Otherwise, `seconds_until_deadline / seconds_to_finish` ∈ (0, 1).

This is a documented stub: the assumption that observed progress rate
will continue is rarely true on real agent runs. Replace the body with a
calibrated predictor before citing the value as a probability anywhere.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .core import Ledger, SubtaskCategory
from .scoring import score


CODING = (SubtaskCategory.PRODUCT, SubtaskCategory.VALIDATION, SubtaskCategory.INVESTIGATION)


def fraction_of_time_to_finish_remaining(
    ledger: Ledger,
    deadline: str | datetime,
    categories: Iterable[SubtaskCategory | str] | None = CODING,
) -> float:
    """Time-budget ratio for reaching progress=1.0 by `deadline`.

    See module docstring. Returns a value in [0.0, 1.0]; this is a time
    ratio, not a probability.
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
