"""
Claim:
JSONL serialization preserves the append-only event log; loading replays that
log to an equivalent score and history.

Plausible wrong implementations:
- Serialize enum reprs instead of stable string values.
- Serialize current state instead of the event log.
- Round-trip through replay with a duplicated init event.
"""

import hashlib
import json
from pathlib import Path

from ledger_progress import EventType, LedgerEvent, Status, apply_event, from_jsonl, new_ledger, replay, score, to_jsonl
from ledger_progress.serialization import event_from_dict, event_to_dict, load_events_jsonl, write_events_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


def event(step, event_type, subtask_id, payload, reason=None):
    return LedgerEvent(step, event_type, subtask_id, payload, reason)


def build_ledger():
    ledger = new_ledger("Fix parser")
    apply_event(ledger, event(1, EventType.ADD_SUBTASK, "S1", {"description": "Locate parser"}, "Plan"))
    apply_event(ledger, event(2, EventType.ADD_EVIDENCE, "S1", {"evidence": ["Parser found."]}))
    apply_event(ledger, event(3, EventType.UPDATE_STATUS, "S1", {"status": "complete"}))
    return ledger


def test_jsonl_round_trip_matches_score_and_does_not_duplicate_init(tmp_path):
    ledger = build_ledger()
    path = tmp_path / "ledger.jsonl"

    to_jsonl(ledger, str(path))
    loaded = from_jsonl(str(path))

    assert score(ledger) == score(loaded)
    assert len(loaded.events) == len(ledger.events)
    assert [event.event_type for event in loaded.events].count(EventType.INIT) == 1


def test_event_json_uses_stable_enum_values():
    event_dict = event_to_dict(event(7, EventType.UPDATE_STATUS, "S1", {"status": "complete"}))
    assert event_dict["event_type"] == "update_status"
    assert json.loads(json.dumps(event_dict)) == event_dict
    assert event_from_dict(event_dict).event_type is EventType.UPDATE_STATUS


def test_event_helpers_preserve_order(tmp_path):
    events = [
        event(0, EventType.INIT, None, {"root_task": "Fix parser"}),
        event(3, EventType.ADD_SUBTASK, "S1", {"description": "Late discovered"}),
        event(2, EventType.ADD_SUBTASK, "S2", {"description": "Earlier step but later log entry"}),
    ]
    path = tmp_path / "events.jsonl"

    write_events_jsonl(events, str(path))

    assert [event.step for event in load_events_jsonl(str(path))] == [0, 3, 2]


def test_canonical_roundtrip_preserves_all_event_fields_and_replay_state(tmp_path):
    source = FIXTURES / "canonical_roundtrip.jsonl"
    raw_lines = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    events = load_events_jsonl(str(source))
    out = tmp_path / "roundtrip.jsonl"

    write_events_jsonl(events, str(out))

    roundtripped_lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    ledger = replay(events)
    roundtripped = from_jsonl(str(out))

    assert roundtripped_lines == raw_lines
    assert [event.step for event in events] == [line["step"] for line in raw_lines]
    assert score(roundtripped) == score(ledger)
    assert roundtripped.subtasks == ledger.subtasks
    assert roundtripped.subtasks["P.1"].parent_id == "P"
    assert roundtripped.subtasks["P.1"].category.value == "product"
    assert roundtripped.subtasks["P.1"].weight == 3.0
    assert roundtripped.subtasks["P.1"].evidence == [
        "parser.py inspected before patch",
        "final_diff.patch shows parser.py modified",
    ]
    assert roundtripped.subtasks["P.2"].status is Status.IN_PROGRESS
    assert roundtripped.subtasks["A.1"].status is Status.INVALIDATED
    assert roundtripped.events[4].reason == "Broad task needed checkable leaves"
    assert roundtripped.events[4].payload["children"][1]["category"] == "validation"


def test_score_does_not_mutate_ledger_jsonl_bytes():
    path = FIXTURES / "canonical_roundtrip.jsonl"
    before = _sha256(path)
    ledger = from_jsonl(str(path))

    score(ledger)
    score(ledger)

    assert _sha256(path) == before


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
