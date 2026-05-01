"""
Claim:
LedgerSession is only an ergonomic event constructor; it must preserve the same
score, replay, and JSONL behavior as explicit LedgerEvent calls.

Plausible wrong implementations:
- Generate different event payloads than the core API expects.
- Hide scoring semantics inside the helper and drift from active-leaf scoring.
- Export a curve per event instead of per ledger step.
"""

import csv

from ledger_progress import (
    EventType,
    LedgerEvent,
    LedgerSession,
    Status,
    SubtaskCategory,
    apply_event,
    from_jsonl,
    new_ledger,
    replay,
    score,
)


def test_session_calls_match_manual_events_score_and_replay():
    session = LedgerSession("Fix parser")
    s1 = session.add("Understand failure", step=1, reason="Plan")
    s2 = session.add("Locate parser", step=1)
    session.start(s1, step=2, evidence="Reading issue")
    session.complete(s1, "Issue restated", step=3)
    session.block(s2, step=3, reason="Need failing test")
    session.reopen(s1, step=4, reason="Restatement missed edge case")
    session.invalidate(s2, step=5, reason="Wrong module")
    children = session.split(s1, ["Add regression", "Patch parser"], step=6, reason="Task was too broad")
    session.complete(children[0], "Regression added", step=7)

    manual = new_ledger("Fix parser")
    for event in [
        LedgerEvent(1, EventType.ADD_SUBTASK, "S1", {"description": "Understand failure", "parent_id": None, "weight": 1.0, "category": SubtaskCategory.PRODUCT}, "Plan"),
        LedgerEvent(1, EventType.ADD_SUBTASK, "S2", {"description": "Locate parser", "parent_id": None, "weight": 1.0, "category": SubtaskCategory.PRODUCT}),
        LedgerEvent(2, EventType.UPDATE_STATUS, "S1", {"status": Status.IN_PROGRESS, "evidence": ["Reading issue"]}),
        LedgerEvent(3, EventType.UPDATE_STATUS, "S1", {"status": Status.COMPLETE, "evidence": ["Issue restated"]}),
        LedgerEvent(3, EventType.UPDATE_STATUS, "S2", {"status": Status.BLOCKED}, "Need failing test"),
        LedgerEvent(4, EventType.REOPEN_SUBTASK, "S1", {"reason": "Restatement missed edge case"}, "Restatement missed edge case"),
        LedgerEvent(5, EventType.INVALIDATE_SUBTASK, "S2", {"reason": "Wrong module"}, "Wrong module"),
        LedgerEvent(6, EventType.SPLIT_SUBTASK, "S1", {"children": [
            {"id": "S1.1", "description": "Add regression"},
            {"id": "S1.2", "description": "Patch parser"},
        ]}, "Task was too broad"),
        LedgerEvent(7, EventType.UPDATE_STATUS, "S1.1", {"status": Status.COMPLETE, "evidence": ["Regression added"]}),
    ]:
        apply_event(manual, event)

    assert score(session.ledger) == score(manual)
    assert score(replay(session.ledger.events)) == score(manual)
    assert session.ledger.events == manual.events


def test_session_jsonl_export_replays_to_same_score(tmp_path):
    session = LedgerSession("Fix parser")
    s1 = session.add("Patch parser", step=1)
    session.complete(s1, "Patch applied", step=2)
    path = tmp_path / "ledger.jsonl"

    session.export_jsonl(str(path))
    loaded = from_jsonl(str(path))

    assert score(loaded) == session.score()
    assert len(loaded.events) == len(session.ledger.events)


def test_session_curve_csv_scores_once_per_step(tmp_path):
    session = LedgerSession("Fix parser")
    s1 = session.add("Understand failure", step=1)
    s2 = session.add("Locate parser", step=1)
    session.complete(s1, "Done", step=2)
    session.complete(s2, "Done", step=2)
    path = tmp_path / "curve.csv"

    session.export_curve_csv(str(path))

    rows = list(csv.DictReader(path.open()))
    assert [row["step"] for row in rows] == ["0", "1", "2"]
    assert [row["progress"] for row in rows] == ["0.0", "0.0", "1.0"]
