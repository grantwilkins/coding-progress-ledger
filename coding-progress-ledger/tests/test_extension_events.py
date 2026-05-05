"""Pass-through behavior for extension event types (e.g. vagrant `state_*` events).

A LedgerEvent with an `event_type` value that is not a member of `EventType`
must be preserved on the events list, must not mutate subtasks, must round-trip
through JSONL, and must not change scoring or queries.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledger_progress import (
    EventType,
    LedgerEvent,
    LedgerSession,
    SubtaskCategory,
    apply_event,
    from_jsonl,
    new_ledger,
    replay,
    score,
    to_jsonl,
)
from ledger_progress.queries import (
    active_incomplete_leaves,
    current_step,
    last_validation_event,
    newly_discovered_since,
    reopens_since,
)
from ledger_progress.serialization import event_from_dict, event_to_dict


EXT = "vagrant.state_read"


def _ledger_with_two_subtasks() -> LedgerSession:
    s = LedgerSession("root")
    s.add("alpha", step=1)
    s.add("beta", step=1)
    return s


# ---- Invariant 1: event preserved ----

def test_extension_event_preserved_in_events():
    s = _ledger_with_two_subtasks()
    apply_event(s.ledger, LedgerEvent(2, EXT, None, {"state_id": "shared", "tokens": 100}))
    types = [e.event_type for e in s.ledger.events]
    assert EXT in types


def test_extension_event_event_type_kept_as_str():
    s = _ledger_with_two_subtasks()
    apply_event(s.ledger, LedgerEvent(2, EXT, None, {"x": 1}))
    ext_events = [e for e in s.ledger.events if not isinstance(e.event_type, EventType)]
    assert len(ext_events) == 1
    assert isinstance(ext_events[0].event_type, str)
    assert ext_events[0].event_type == EXT


def test_known_event_type_string_still_coerces_to_enum():
    """Strings that match an EventType value must still coerce, for backwards compat."""
    s = _ledger_with_two_subtasks()
    e = LedgerEvent(2, "update_status", "S1", {"status": "complete", "evidence": ["done"]})
    assert e.event_type is EventType.UPDATE_STATUS


# ---- Invariant 2: no subtask mutation ----

def test_extension_event_does_not_mutate_subtasks():
    s = _ledger_with_two_subtasks()
    before = {sid: (st.status, st.updated_at_step, list(st.evidence))
              for sid, st in s.ledger.subtasks.items()}
    apply_event(s.ledger, LedgerEvent(2, EXT, "S1", {"state_id": "shared"}))
    after = {sid: (st.status, st.updated_at_step, list(st.evidence))
             for sid, st in s.ledger.subtasks.items()}
    assert before == after


def test_extension_event_with_unknown_subtask_id_does_not_raise():
    """Pass-through must not validate subtask_id, since extension semantics own it."""
    s = _ledger_with_two_subtasks()
    apply_event(s.ledger, LedgerEvent(2, EXT, "Snonexistent", {}))


# ---- Invariant 3: JSONL round-trip ----

def test_extension_event_round_trips_through_dict():
    e = LedgerEvent(3, EXT, None, {"state_id": "x", "tokens": 5})
    d = event_to_dict(e)
    assert d["event_type"] == EXT
    e2 = event_from_dict(d)
    assert e2.event_type == EXT
    assert e2.payload == {"state_id": "x", "tokens": 5}


def test_extension_event_round_trips_through_jsonl(tmp_path: Path):
    s = _ledger_with_two_subtasks()
    apply_event(s.ledger, LedgerEvent(2, EXT, None, {"state_id": "shared", "tokens": 7}))
    apply_event(s.ledger, LedgerEvent(2, "vagrant.state_write", None, {"state_id": "out"}))
    apply_event(s.ledger, LedgerEvent(3, EventType.UPDATE_STATUS, "S1",
                                      {"status": "complete", "evidence": ["done"]}))

    path = str(tmp_path / "trace.jsonl")
    to_jsonl(s.ledger, path)
    restored = from_jsonl(path)

    original_types = [e.event_type for e in s.ledger.events]
    restored_types = [e.event_type for e in restored.events]
    assert original_types == restored_types

    original_dicts = [event_to_dict(e) for e in s.ledger.events]
    restored_dicts = [event_to_dict(e) for e in restored.events]
    assert original_dicts == restored_dicts


def test_extension_event_jsonl_format_is_plain_string():
    """Ensure the wire format stores the event_type as the literal extension string."""
    e = LedgerEvent(2, EXT, None, {"x": 1})
    d = event_to_dict(e)
    assert json.dumps(d)
    assert d["event_type"] == EXT


# ---- Invariant 4: scoring and queries unchanged ----

def test_scoring_unchanged_by_extension_events():
    s_clean = LedgerSession("root")
    s_clean.add("alpha", step=1)
    s_clean.add("beta", step=1)
    s_clean.complete("S1", step=2, evidence=["done"])
    clean_score = score(s_clean.ledger)

    s_ext = LedgerSession("root")
    s_ext.add("alpha", step=1)
    s_ext.add("beta", step=1)
    apply_event(s_ext.ledger, LedgerEvent(1, EXT, None, {"state_id": "x", "tokens": 100}))
    apply_event(s_ext.ledger, LedgerEvent(1, "vagrant.placement_decision", None,
                                          {"node_id": "S1", "site": "phoenix"}))
    s_ext.complete("S1", step=2, evidence=["done"])
    apply_event(s_ext.ledger, LedgerEvent(2, "vagrant.state_invalidate", None, {"state_id": "x"}))
    ext_score = score(s_ext.ledger)

    assert clean_score.complete_weight == ext_score.complete_weight
    assert clean_score.active_weight == ext_score.active_weight
    assert clean_score.progress == ext_score.progress
    assert clean_score.complete_leaf_count == ext_score.complete_leaf_count
    assert clean_score.active_leaf_count == ext_score.active_leaf_count


def test_active_incomplete_leaves_ignores_extension_events():
    s = _ledger_with_two_subtasks()
    leaves_before = {l.id for l in active_incomplete_leaves(s.ledger)}
    apply_event(s.ledger, LedgerEvent(1, EXT, None, {"state_id": "x"}))
    leaves_after = {l.id for l in active_incomplete_leaves(s.ledger)}
    assert leaves_before == leaves_after


def test_reopens_since_excludes_extension_events():
    s = _ledger_with_two_subtasks()
    s.complete("S1", step=2, evidence=["done"])
    s.reopen("S1", step=3, reason="regression")
    apply_event(s.ledger, LedgerEvent(4, EXT, None, {}))
    reopens = reopens_since(s.ledger, step=0)
    assert len(reopens) == 1
    assert reopens[0].event_type is EventType.REOPEN_SUBTASK


def test_current_step_includes_extension_events():
    s = _ledger_with_two_subtasks()
    apply_event(s.ledger, LedgerEvent(99, EXT, None, {}))
    assert current_step(s.ledger) == 99


def test_last_validation_event_unchanged_by_extension():
    s = LedgerSession("root")
    s.add("test_alpha", step=1, category=SubtaskCategory.VALIDATION)
    s.complete("S1", step=2, evidence=["passed"])
    apply_event(s.ledger, LedgerEvent(3, EXT, None, {"state_id": "x"}))
    last = last_validation_event(s.ledger)
    assert last is not None
    assert last.event_type is EventType.UPDATE_STATUS


# ---- Negative checks: malformed extension events still hard-fail ----

def test_empty_event_type_string_rejected():
    with pytest.raises(ValueError):
        LedgerEvent(1, "", None, {})


def test_non_string_non_enum_event_type_rejected():
    with pytest.raises(ValueError):
        LedgerEvent(1, 42, None, {})


def test_replay_still_requires_init_first():
    """An extension event as the first event must not be accepted as INIT."""
    events = [LedgerEvent(0, EXT, None, {})]
    with pytest.raises(ValueError, match="init"):
        replay(events)


def test_replay_with_extension_events_after_init():
    """Replay over a sequence with extension events sprinkled in produces an equivalent ledger."""
    s = LedgerSession("root")
    s.add("alpha", step=1)
    apply_event(s.ledger, LedgerEvent(1, EXT, None, {"state_id": "x"}))
    s.complete("S1", step=2, evidence=["done"])
    apply_event(s.ledger, LedgerEvent(3, "vagrant.migration_end", None, {"elapsed_s": 1.2}))

    rebuilt = replay(list(s.ledger.events))
    assert [e.event_type for e in rebuilt.events] == [e.event_type for e in s.ledger.events]
    assert rebuilt.subtasks["S1"].status.value == "complete"


def test_apply_init_after_first_event_rejected():
    """An INIT event after the first event is still rejected (regression guard)."""
    s = new_ledger("root")
    with pytest.raises(ValueError, match="init"):
        apply_event(s, LedgerEvent(1, EventType.INIT, None, {"root_task": "root"}))
