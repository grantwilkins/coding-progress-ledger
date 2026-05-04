"""Prefix replay engine.

Given a RunRecord and a step `t`, return a `ReplayState` that contains
ONLY events with step <= t and the upstream ledger state derived from
those events. This is the most leakage-sensitive function in the
project: a bug here is invisible to every downstream test that uses
prefix-only inputs.

Invariants enforced at runtime (not just by tests):
- `events_so_far` contains only events with step <= t_step.
- The replayed `Ledger` is the upstream `replay()` of `events_so_far`,
  so we inherit upstream's schema validation.
- `prefix_replay` is a pure function of (RunRecord, t_step). Calling
  twice on the same inputs produces equal output. Future events past
  t_step have no observable influence.
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger_progress.core import Ledger, LedgerEvent, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import ProgressObservation, score

from coding_estimator.ingest.run_record import RunRecord


class FutureLeakageError(RuntimeError):
    """Raised when an event with step > t_step is found in the replayed
    prefix. This means a caller corrupted the run, or the replay engine
    has a leakage bug."""


@dataclass(frozen=True)
class ReplayState:
    t_step: int
    events_so_far: tuple[LedgerEvent, ...]
    ledger: Ledger
    coding_score: ProgressObservation


def _assert_prefix_only(events: tuple[LedgerEvent, ...], t_step: int) -> None:
    for e in events:
        if e.step > t_step:
            raise FutureLeakageError(
                f"event with step={e.step} > t_step={t_step} present in prefix; "
                f"event_type={e.event_type.value} subtask_id={e.subtask_id}"
            )


def prefix_replay(run: RunRecord, t_step: int) -> ReplayState:
    """Return the replay state at step `t_step`. All events with step
    <= t_step are included; everything past is excluded."""
    if t_step < 0:
        raise ValueError(f"t_step must be non-negative, got {t_step}")
    events_so_far = tuple(e for e in run.events if e.step <= t_step)
    _assert_prefix_only(events_so_far, t_step)
    ledger = replay(list(events_so_far))
    obs = score(ledger, categories=CODING_CATEGORIES)
    return ReplayState(
        t_step=t_step,
        events_so_far=events_so_far,
        ledger=ledger,
        coding_score=obs,
    )
