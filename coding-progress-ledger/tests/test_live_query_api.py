"""Live query API: the 'check and query' verb of the user's mission.

The five queries (current_step, active_blocked_leaves, reopens_since,
newly_discovered_since, last_validation_event, stalled_for) must
correctly answer "what is the current state of run X?" without
re-reading CSV files.
"""
from __future__ import annotations

from ledger_progress import LedgerSession, Status, SubtaskCategory
from ledger_progress.queries import (
    active_blocked_leaves,
    current_step,
    last_validation_event,
    newly_discovered_since,
    reopens_since,
    stalled_for,
)


def _session() -> LedgerSession:
    s = LedgerSession("Fix bug", clock=lambda: None)
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    s.complete("S1", "step 5: localized at file.py:42", step=5)
    s.add("Patch", step=6, category=SubtaskCategory.PRODUCT)
    s.add("Validate", step=7, category=SubtaskCategory.VALIDATION)
    s.block("S2", step=10, reason="Stuck loop", evidence="step 10: third edit retry rejected")
    return s


def test_current_step_returns_max_event_step():
    assert current_step(_session().ledger) == 10


def test_active_blocked_leaves_only_returns_blocked():
    leaves = active_blocked_leaves(_session().ledger)
    assert [s.id for s in leaves] == ["S2"]


def test_reopens_since_excludes_pre_threshold():
    s = _session()
    s.reopen("S1", step=12, reason="Re-investigation needed")
    assert len(reopens_since(s.ledger, step=11)) == 1
    assert reopens_since(s.ledger, step=12) == []


def test_newly_discovered_since_window():
    discovered = newly_discovered_since(_session().ledger, step=6)
    assert [s.id for s in discovered] == ["S3"]


def test_last_validation_event_finds_latest():
    s = _session()
    s.complete("S3", "step 15: pytest passed", step=15)
    last = last_validation_event(s.ledger)
    assert last is not None and last.step == 15 and last.subtask_id == "S3"


def test_last_validation_event_none_when_no_val_leaf():
    s = LedgerSession("Fix bug", clock=lambda: None)
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    assert last_validation_event(s.ledger) is None


def test_stalled_for_counts_steps_since_block():
    s = _session()
    s.add("More work", step=18, category=SubtaskCategory.PRODUCT)
    assert stalled_for(s.ledger, status=Status.BLOCKED) == 18 - 10


def test_stalled_for_zero_when_nothing_blocked():
    s = LedgerSession("Fix bug", clock=lambda: None)
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    assert stalled_for(s.ledger) == 0
