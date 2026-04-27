"""
Claim:
Progress is complete active leaf weight divided by total active leaf weight.

Plausible wrong implementations:
- Score subtasks by count instead of weight.
- Count parent nodes even when active children exist.
- Treat invalidated or deleted children as active children.
- Allow reverse-progress events without changing the denominator or numerator.
- Leave partial state behind after rejecting an invalid completion event.
"""

import pytest

from conftest import (
    add_subtask,
    event,
    make_ledger_with_eight_subtasks_two_complete,
    make_ledger_with_four_subtasks,
    mark_complete,
    reopen_subtask,
    split_subtask,
)
from ledger_progress import EventType, Status, apply_event, new_ledger, score


def test_empty_ledger_scores_zero():
    obs = score(new_ledger("Fix bug"))
    assert obs.progress == 0.0
    assert obs.active_weight == 0.0
    assert obs.complete_weight == 0.0


def test_progress_increases_when_work_completed():
    ledger = make_ledger_with_four_subtasks()
    ledger = mark_complete(ledger, "S1", evidence="Issue behavior restated", step=2)
    ledger = mark_complete(ledger, "S2", evidence="Opened relevant parser file", step=3)

    obs = score(ledger)
    assert obs.complete_weight == 2.0
    assert obs.active_weight == 4.0
    assert obs.progress == 0.5


def test_reverse_progress_when_new_work_is_discovered():
    ledger = make_ledger_with_four_subtasks()
    ledger = mark_complete(ledger, "S1", evidence="done", step=2)
    ledger = mark_complete(ledger, "S2", evidence="done", step=3)
    assert score(ledger).progress == 0.5

    for description in [
        "Fix serializer regression",
        "Add serializer test",
        "Run broader test suite",
        "Update docs or final explanation",
    ]:
        ledger = add_subtask(ledger, description, step=4)

    obs = score(ledger)
    assert obs.complete_weight == 2.0
    assert obs.active_weight == 8.0
    assert obs.progress == 0.25


def test_reopening_completed_work_decreases_progress():
    ledger = make_ledger_with_eight_subtasks_two_complete()
    assert score(ledger).progress == 0.25

    ledger = reopen_subtask(ledger, "S2", reason="The file was relevant but not the root cause", step=5)

    obs = score(ledger)
    assert obs.complete_weight == 1.0
    assert obs.active_weight == 8.0
    assert obs.progress == 0.125


def test_complete_requires_evidence():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "Patch parser", step=1)
    before_events = len(ledger.events)

    with pytest.raises(ValueError):
        mark_complete(ledger, "S1", evidence="", step=2)

    assert ledger.subtasks["S1"].status is Status.NOT_STARTED
    assert len(ledger.events) == before_events


def test_parent_with_children_does_not_count_as_leaf():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "Validate fix", step=1)
    ledger = split_subtask(ledger, "S1", [
        "Run targeted parser test",
        "Run broader date test suite",
        "Check no formatting regressions",
    ], step=2)

    obs = score(ledger)
    assert obs.active_leaf_count == 3
    assert obs.active_weight == 3.0


def test_splitting_completed_parent_can_reduce_progress():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "Validate fix", step=1)
    ledger = mark_complete(ledger, "S1", evidence="One targeted test passed", step=2)
    assert score(ledger).progress == 1.0

    ledger = split_subtask(ledger, "S1", [
        "Run targeted parser test",
        "Run broader date test suite",
        "Check no formatting regressions",
    ], step=3, reason="Validation was underspecified")

    obs = score(ledger)
    assert obs.progress == 0.0
    assert obs.active_leaf_count == 3


def test_invalidated_and_deleted_subtasks_not_counted_but_kept_in_history():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "First path", step=1)
    ledger = add_subtask(ledger, "Duplicate task", step=1)
    ledger = mark_complete(ledger, "S1", evidence="done", step=2)

    apply_event(ledger, event(3, EventType.INVALIDATE_SUBTASK, "S1", {"reason": "Wrong path"}))
    obs = score(ledger)
    assert "S1" in ledger.subtasks
    assert ledger.events[-1].event_type is EventType.INVALIDATE_SUBTASK
    assert (obs.complete_weight, obs.active_weight, obs.progress) == (0, 1, 0.0)

    apply_event(ledger, event(4, EventType.DELETE_SUBTASK, "S2", {"reason": "Duplicate"}))
    obs = score(ledger)
    assert "S2" in ledger.subtasks
    assert ledger.events[-1].event_type is EventType.DELETE_SUBTASK
    assert (obs.complete_weight, obs.active_weight, obs.progress) == (0, 0, 0.0)


def test_inactive_children_do_not_block_parent_leaf_status():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "Parent", step=1, subtask_id="P", weight=10)
    ledger = mark_complete(ledger, "P", evidence="Parent was done", step=2)
    ledger = split_subtask(ledger, "P", [
        {"id": "C1", "description": "Child one", "weight": 2},
        {"id": "C2", "description": "Child two", "weight": 3},
    ], step=3)

    assert (score(ledger).complete_weight, score(ledger).active_weight) == (0, 5)

    apply_event(ledger, event(4, EventType.INVALIDATE_SUBTASK, "C1", {"reason": "No longer needed"}))
    apply_event(ledger, event(5, EventType.DELETE_SUBTASK, "C2", {"reason": "Duplicate"}))
    obs = score(ledger)
    assert (obs.complete_weight, obs.active_weight, obs.progress) == (10, 10, 1.0)


def test_weighted_scoring_is_not_count_based():
    ledger = new_ledger("Fix bug")
    ledger = add_subtask(ledger, "Heavy task", step=1, subtask_id="Heavy", weight=5)
    ledger = add_subtask(ledger, "Light task", step=1, subtask_id="Light", weight=1)
    ledger = mark_complete(ledger, "Heavy", evidence="Heavy task done", step=2)

    obs = score(ledger)
    assert (obs.complete_weight, obs.active_weight, obs.progress) == (5, 6, 5 / 6)
