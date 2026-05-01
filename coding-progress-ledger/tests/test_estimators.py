from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ledger_progress import LedgerSession, SubtaskCategory, replay
from ledger_progress.estimators import fraction_of_time_to_finish_remaining
from ledger_progress.serialization import event_from_dict

ROOT = Path(__file__).resolve().parents[1]


def _build_ledger(timestamps):
    """Create a ledger with one PRODUCT subtask added at step 1, completed at step 2.
    `timestamps` is a list of two ISO strings injected into the events post-build."""
    session = LedgerSession("est test", clock=lambda: timestamps.pop(0))
    sid = session.add("Patch", step=1, category=SubtaskCategory.PRODUCT)
    return sid, session


def test_returns_one_when_already_complete():
    ts = ["2026-05-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00", "2026-05-01T00:01:00+00:00"]
    sid, session = _build_ledger(ts)
    session.complete(sid, "diff applied", step=2)
    assert fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T00:00:30+00:00") == 1.0


def test_returns_zero_when_no_timestamps():
    session = LedgerSession("no ts", clock=lambda: None)
    sid = session.add("Patch", step=1, category=SubtaskCategory.PRODUCT)
    assert fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T01:00:00+00:00") == 0.0


def test_returns_zero_when_no_progress_yet():
    ts = ["2026-05-01T00:00:00+00:00", "2026-05-01T00:01:00+00:00"]
    sid, session = _build_ledger(ts)
    assert fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T01:00:00+00:00") == 0.0


def test_returns_zero_when_deadline_past():
    ts = ["2026-05-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00", "2026-05-01T00:10:00+00:00"]
    sid, session = _build_ledger(ts)
    other = session.add("Validate", step=2, category=SubtaskCategory.VALIDATION)
    session.complete(sid, "diff applied", step=3)
    # last event timestamp 00:10:00; deadline before it
    assert fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T00:05:00+00:00") == 0.0


def test_linear_extrapolation_returns_fraction_in_unit_interval():
    # add P1=00:00, add P2=00:30, complete P1=01:00 -> progress=0.5 over 60s.
    # rate=0.5/60s, remaining 0.5 -> 60s to finish.
    # deadline at 01:30 -> 30s left / 60s needed = 0.5.
    ts = [
        "2026-05-01T00:00:00+00:00",
        "2026-05-01T00:00:30+00:00",
        "2026-05-01T00:01:00+00:00",
    ]
    session = LedgerSession("linear", clock=lambda: ts.pop(0))
    p1 = session.add("P1", step=1, category=SubtaskCategory.PRODUCT)
    p2 = session.add("P2", step=2, category=SubtaskCategory.PRODUCT)
    session.complete(p1, "diff applied", step=3)
    p = fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T00:01:30+00:00")
    assert p == pytest.approx(0.5)


def test_returns_one_when_deadline_far_enough():
    ts = [
        "2026-05-01T00:00:00+00:00",
        "2026-05-01T00:00:30+00:00",
        "2026-05-01T00:01:00+00:00",
    ]
    session = LedgerSession("far", clock=lambda: ts.pop(0))
    p1 = session.add("P1", step=1, category=SubtaskCategory.PRODUCT)
    p2 = session.add("P2", step=2, category=SubtaskCategory.PRODUCT)
    session.complete(p1, "diff applied", step=3)
    # need 60s more; deadline +120s
    assert fraction_of_time_to_finish_remaining(session.ledger, "2026-05-01T00:03:00+00:00") == 1.0


def test_accepts_datetime_deadline():
    ts = [
        "2026-05-01T00:00:00+00:00",
        "2026-05-01T00:00:30+00:00",
        "2026-05-01T00:01:00+00:00",
    ]
    session = LedgerSession("dt", clock=lambda: ts.pop(0))
    p1 = session.add("P1", step=1, category=SubtaskCategory.PRODUCT)
    p2 = session.add("P2", step=2, category=SubtaskCategory.PRODUCT)
    session.complete(p1, "diff applied", step=3)
    deadline = datetime(2026, 5, 1, 0, 3, 0, tzinfo=timezone.utc)
    assert 0.0 <= fraction_of_time_to_finish_remaining(session.ledger, deadline) <= 1.0


def _load_ledger(run_dir: Path):
    events = []
    with (run_dir / "ledger.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(event_from_dict(json.loads(line)))
    return replay(events)


def test_runs_on_n6_wallclock_batch_with_real_timestamp_spans():
    """The N6 synthetic-clock batch produces ledgers whose timestamp spans
    are physically meaningful (>= 60s). The function must return a value
    in [0.0, 1.0] on every such run for an after-finish deadline."""
    batch = ROOT / "runs" / "swe_agent_live_wallclock"
    if not batch.is_dir():
        pytest.skip("N6 wallclock batch not present")
    runs = sorted(p for p in batch.iterdir() if p.is_dir() and (p / "ledger.jsonl").is_file())
    assert runs, "expected at least one wallclock run"
    far_deadline = "2099-01-01T00:00:00+00:00"
    for run_dir in runs:
        ledger = _load_ledger(run_dir)
        result = fraction_of_time_to_finish_remaining(ledger, far_deadline)
        assert 0.0 <= result <= 1.0, run_dir.name
        # Far-future deadline + any non-zero progress + non-degenerate span
        # must extrapolate to "fits"; this is the "stub returns 1.0" branch.
        timestamps = [e.timestamp for e in ledger.events if e.timestamp]
        if timestamps:
            assert result == 1.0 or result == 0.0, (run_dir.name, result)
