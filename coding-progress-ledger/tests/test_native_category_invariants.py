"""Workstream J invariants: J1 (session always emits category) and
J2 (check_native_categories enforcement). Locks in the claims of
runs/swe_agent_pilot/CATEGORY_RESOLUTION_REPORT.md.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger_progress import LedgerSession, SubtaskCategory, from_jsonl  # noqa: E402
from ledger_progress.core import EventType  # noqa: E402
from scripts.check_native_categories import offending_events, scan  # noqa: E402


def test_add_emits_category_in_payload_for_every_call():
    # Catches the pre-J1 "PRODUCT is default — strip it" optimization regressing.
    session = LedgerSession("Root")
    session.add("a", step=1, category=SubtaskCategory.VALIDATION)
    session.add("b", step=1, category=SubtaskCategory.INVESTIGATION)
    session.add("c", step=1, category=SubtaskCategory.DOCUMENTATION)
    session.add("d", step=1)
    adds = [e for e in session.ledger.events if e.event_type is EventType.ADD_SUBTASK]
    assert len(adds) == 4
    for event in adds:
        assert "category" in event.payload
    assert adds[-1].payload["category"] is SubtaskCategory.PRODUCT


def test_explicit_product_roundtrips_through_jsonl(tmp_path):
    # Catches the silent-strip regression: serialized line must contain "category":"product".
    session = LedgerSession("Root")
    session.add("only", step=1, category=SubtaskCategory.PRODUCT)
    path = tmp_path / "ledger.jsonl"
    session.export_jsonl(str(path))
    text = path.read_text()
    assert '"category":"product"' in text
    loaded = from_jsonl(str(path))
    assert loaded.subtasks["S1"].category is SubtaskCategory.PRODUCT


def test_offending_events_empty_on_fully_native_ledger(tmp_path):
    # Round-trip sanity: a session-built ledger must satisfy J2's enforcement.
    session = LedgerSession("Root")
    s1 = session.add("p", step=1, category=SubtaskCategory.PRODUCT)
    session.add("v", step=1, category=SubtaskCategory.VALIDATION)
    session.start(s1, step=2)
    session.split(s1, ["c1", "c2"], step=3, reason="too broad",
                  categories=[SubtaskCategory.PRODUCT, SubtaskCategory.VALIDATION])
    path = tmp_path / "ledger.jsonl"
    session.export_jsonl(str(path))
    assert offending_events(path) == []


def test_offending_events_flags_add_without_category(tmp_path):
    # Pre-J1 corpus shape: ADD_SUBTASK payload missing category must be flagged exactly once.
    path = tmp_path / "ledger.jsonl"
    lines = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"description": "Root"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "x", "parent_id": None, "weight": 1.0}, "reason": None},
    ]
    path.write_text("".join(json.dumps(l) + "\n" for l in lines))
    offenders = offending_events(path)
    assert len(offenders) == 1
    assert offenders[0]["event_type"] == "add_subtask"
    assert offenders[0]["subtask_id"] == "S1"
    assert offenders[0]["child_id"] is None


def test_offending_events_flags_split_child_without_category(tmp_path):
    # Future regression: split children must each carry category; only the missing one is flagged.
    path = tmp_path / "ledger.jsonl"
    lines = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"description": "Root"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "x", "parent_id": None, "weight": 1.0, "category": "product"},
         "reason": None},
        {"step": 2, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "in_progress"}, "reason": None},
        {"step": 3, "event_type": "split_subtask", "subtask_id": "S1",
         "payload": {"children": [
             {"id": "S1.1", "description": "a", "category": "product"},
             {"id": "S1.2", "description": "b"},
         ]}, "reason": "split"},
    ]
    path.write_text("".join(json.dumps(l) + "\n" for l in lines))
    offenders = offending_events(path)
    assert len(offenders) == 1
    assert offenders[0]["event_type"] == "split_subtask"
    assert offenders[0]["child_id"] == "S1.2"


def test_swe_agent_corpus_has_zero_violations():
    # Catches an on-disk J1 revert: every shipped pilot ledger must be native.
    roots = [ROOT / "runs" / "swe_agent_pilot"]
    v3 = ROOT / "runs" / "swe_agent_pilot_v3"
    if v3.exists():
        roots.append(v3)
    report = scan(roots, set())
    assert report["violating_runs"] == {}
