"""
Claim:
LedgerSetSession mirrors LedgerSession's ergonomics one level up:
add_member appends a member with auto-assigned member_id, mark_member sets
status_override, score() returns score_set, export_jsonl round-trips.

Plausible wrong implementations:
- Reuse member_id when an earlier member was marked-overridden.
- Silently no-op mark_member on unknown member_id.
- export_jsonl drops mark_member events.
"""

from pathlib import Path

import pytest

from ledger_progress import (
    LedgerSession,
    LedgerSetSession,
    Status,
    read_set_jsonl,
)


def _write_ledger(tmp_path: Path, name: str, coding_progress: float) -> str:
    session = LedgerSession(f"task {name}", clock=lambda: None)
    ids = [session.add(f"leaf {i}", step=1) for i in range(4)]
    n_complete = round(coding_progress * 4)
    for i in range(n_complete):
        session.complete(ids[i], "done", step=2)
    path = tmp_path / f"{name}.jsonl"
    session.export_jsonl(str(path))
    return str(path)


def test_add_member_assigns_sequential_ids_and_score_matches(tmp_path):
    a = _write_ledger(tmp_path, "a", 1.00)
    b = _write_ledger(tmp_path, "b", 0.50)

    session = LedgerSetSession("rollup")
    m1 = session.add_member(a, weight=1.0)
    m2 = session.add_member(b, weight=3.0)

    assert m1 == "M1"
    assert m2 == "M2"
    assert session.score() == (1.0 * 1.00 + 3.0 * 0.50) / 4.0


def test_explicit_member_id_honored_and_duplicates_rejected(tmp_path):
    a = _write_ledger(tmp_path, "a", 1.00)

    session = LedgerSetSession("rollup")
    session.add_member(a, member_id="pilot_s_01")

    with pytest.raises(ValueError, match="duplicate member_id"):
        session.add_member(a, member_id="pilot_s_01")


def test_mark_member_sets_status_override(tmp_path):
    a = _write_ledger(tmp_path, "a", 1.00)
    b = _write_ledger(tmp_path, "b", 0.00)

    session = LedgerSetSession("rollup")
    session.add_member(a)
    session.add_member(b)
    session.mark_member("M2", Status.INVALIDATED)

    assert session.set.members[1].status_override is Status.INVALIDATED
    assert session.score() == 1.0


def test_mark_member_unknown_id_raises(tmp_path):
    session = LedgerSetSession("x")

    with pytest.raises(ValueError, match="unknown member_id"):
        session.mark_member("M99", Status.INVALIDATED)


def test_export_jsonl_round_trip_preserves_overrides(tmp_path):
    a = _write_ledger(tmp_path, "a", 1.00)
    b = _write_ledger(tmp_path, "b", 0.50)

    session = LedgerSetSession("rollup")
    session.add_member(a, weight=2.0)
    session.add_member(b, weight=1.0)
    session.mark_member("M2", Status.DELETED)

    out = tmp_path / "set.jsonl"
    session.export_jsonl(str(out))
    loaded = read_set_jsonl(str(out))

    assert loaded == session.set
    assert loaded.members[1].status_override is Status.DELETED
