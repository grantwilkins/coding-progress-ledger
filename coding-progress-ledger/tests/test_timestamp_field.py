"""LedgerEvent timestamp: optional ISO-8601 wall-clock; unlocks deadline modeling.

Long-range critic established that 'p(finish-by-deadline)' is undefined
without wall-clock. These tests lock in:
- timestamp survives serialization round-trip
- LedgerSession auto-stamps when no clock override
- clock=lambda: None disables auto-stamp (back-compat for replay-equality tests)
- legacy ledger.jsonl without timestamps still loads (None is the default)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ledger_progress import LedgerSession, SubtaskCategory
from ledger_progress.core import EventType, LedgerEvent
from ledger_progress.serialization import event_from_dict, event_to_dict, from_jsonl, to_jsonl


def test_timestamp_survives_roundtrip(tmp_path):
    s = LedgerSession("Fix bug")
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    s.complete("S1", "step 2: localized", step=2)
    path = tmp_path / "ledger.jsonl"
    to_jsonl(s.ledger, str(path))
    loaded = from_jsonl(str(path))
    for original, restored in zip(s.ledger.events, loaded.events):
        assert original.timestamp == restored.timestamp


def test_session_default_clock_emits_iso_utc():
    s = LedgerSession("Fix bug")
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    add_event = s.ledger.events[1]
    assert add_event.timestamp is not None
    parsed = datetime.fromisoformat(add_event.timestamp)
    assert parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_clock_override_can_disable_timestamps():
    s = LedgerSession("Fix bug", clock=lambda: None)
    s.add("Investigate", step=1, category=SubtaskCategory.INVESTIGATION)
    assert all(e.timestamp is None for e in s.ledger.events[1:])


def test_legacy_event_dict_without_timestamp_loads():
    legacy = {
        "step": 5,
        "event_type": "add_subtask",
        "subtask_id": "S1",
        "payload": {"description": "x", "parent_id": None, "weight": 1.0, "category": "product"},
        "reason": None,
    }
    event = event_from_dict(legacy)
    assert event.timestamp is None


def test_event_to_dict_omits_none_timestamp():
    event = LedgerEvent(0, EventType.INIT, None, {"root_task": "x"})
    d = event_to_dict(event)
    assert "timestamp" not in d


def test_event_to_dict_includes_set_timestamp():
    event = LedgerEvent(0, EventType.INIT, None, {"root_task": "x"}, None, "2026-04-30T12:00:00+00:00")
    d = event_to_dict(event)
    assert d["timestamp"] == "2026-04-30T12:00:00+00:00"
