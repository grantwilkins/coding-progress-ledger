from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ledger_progress import LedgerSession, SubtaskCategory
from ledger_progress.estimators import p_finish_by


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
    assert p_finish_by(session.ledger, "2026-05-01T00:00:30+00:00") == 1.0


def test_returns_zero_when_no_timestamps():
    session = LedgerSession("no ts", clock=lambda: None)
    sid = session.add("Patch", step=1, category=SubtaskCategory.PRODUCT)
    assert p_finish_by(session.ledger, "2026-05-01T01:00:00+00:00") == 0.0


def test_returns_zero_when_no_progress_yet():
    ts = ["2026-05-01T00:00:00+00:00", "2026-05-01T00:01:00+00:00"]
    sid, session = _build_ledger(ts)
    assert p_finish_by(session.ledger, "2026-05-01T01:00:00+00:00") == 0.0


def test_returns_zero_when_deadline_past():
    ts = ["2026-05-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00", "2026-05-01T00:10:00+00:00"]
    sid, session = _build_ledger(ts)
    other = session.add("Validate", step=2, category=SubtaskCategory.VALIDATION)
    session.complete(sid, "diff applied", step=3)
    # last event timestamp 00:10:00; deadline before it
    assert p_finish_by(session.ledger, "2026-05-01T00:05:00+00:00") == 0.0


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
    p = p_finish_by(session.ledger, "2026-05-01T00:01:30+00:00")
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
    assert p_finish_by(session.ledger, "2026-05-01T00:03:00+00:00") == 1.0


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
    assert 0.0 <= p_finish_by(session.ledger, deadline) <= 1.0
