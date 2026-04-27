"""
Claim:
The event log is the source of truth and replay reconstructs the same ledger
state and score without duplicating the init event.

Plausible wrong implementations:
- Replaying through new_ledger duplicates init.
- Replay applies events out of order.
- Replay reconstructs current state but loses history.
"""

import pytest

from ledger_progress import EventType, LedgerEvent, apply_event, new_ledger, replay, score


def event(step, event_type, subtask_id, payload):
    return LedgerEvent(step, event_type, subtask_id, payload)


def test_replay_matches_original_score_and_history():
    original = new_ledger("Fix parser")
    apply_event(original, event(1, EventType.ADD_SUBTASK, "S1", {"description": "Locate parser"}))
    apply_event(original, event(2, EventType.UPDATE_STATUS, "S1", {
        "status": "complete",
        "evidence": ["Found parser."],
    }))
    apply_event(original, event(3, EventType.SPLIT_SUBTASK, "S1", {"children": [
        {"id": "S1.1", "description": "Add regression", "weight": 2},
        {"id": "S1.2", "description": "Run tests", "weight": 1},
    ]}))
    apply_event(original, event(4, EventType.UPDATE_STATUS, "S1.1", {
        "status": "complete",
        "evidence": ["Regression added."],
    }))

    replayed = replay(original.events)

    assert score(original) == score(replayed)
    assert len(replayed.events) == len(original.events)
    assert [event.event_type for event in replayed.events].count(EventType.INIT) == 1
    assert replayed.subtasks == original.subtasks


def test_replay_requires_init_first_event():
    with pytest.raises(ValueError):
        replay([event(1, EventType.ADD_SUBTASK, "S1", {"description": "No init"})])
