import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress import EventType, LedgerEvent, apply_event, new_ledger, replay, score


def event(step, event_type, subtask_id, payload=None, reason=None):
    return LedgerEvent(step, event_type, subtask_id, payload or {}, reason)


def add_subtask(ledger, description, step, subtask_id=None, **payload):
    subtask_id = subtask_id or f"S{len(ledger.subtasks) + 1}"
    return apply_event(ledger, event(step, EventType.ADD_SUBTASK, subtask_id, {
        "description": description,
        **payload,
    }))


def mark_complete(ledger, subtask_id, evidence, step):
    return apply_event(ledger, event(step, EventType.UPDATE_STATUS, subtask_id, {
        "status": "complete",
        "evidence": [evidence],
    }))


def reopen_subtask(ledger, subtask_id, reason, step):
    return apply_event(ledger, event(step, EventType.REOPEN_SUBTASK, subtask_id, {"reason": reason}, reason))


def split_subtask(ledger, subtask_id, children, step, reason=None):
    return apply_event(ledger, event(step, EventType.SPLIT_SUBTASK, subtask_id, {
        "children": [
            child if isinstance(child, dict) else {"id": f"{subtask_id}.{i}", "description": child}
            for i, child in enumerate(children, 1)
        ],
    }, reason))


def make_ledger_with_four_subtasks():
    ledger = new_ledger("Fix bug")
    for description in ["Understand failure", "Locate code", "Patch code", "Run tests"]:
        ledger = add_subtask(ledger, description, step=1)
    return ledger


def make_ledger_with_eight_subtasks_two_complete():
    ledger = make_ledger_with_four_subtasks()
    ledger = mark_complete(ledger, "S1", evidence="done", step=2)
    ledger = mark_complete(ledger, "S2", evidence="done", step=3)
    for description in [
        "Fix serializer regression",
        "Add serializer test",
        "Run broader test suite",
        "Update docs or final explanation",
    ]:
        ledger = add_subtask(ledger, description, step=4)
    return ledger


def replay_progress_curve(events):
    ledger = replay(events[:1])
    curve = [_row(ledger)]
    for index, event_ in enumerate(events[1:], 1):
        apply_event(ledger, event_)
        if index == len(events) - 1 or events[index + 1].step != event_.step:
            curve.append(_row(ledger))
    return curve


def _row(ledger):
    obs = score(ledger)
    return (obs.step, obs.complete_weight, obs.active_weight, obs.progress)
