"""
Claim:
LedgerSet round-trips through set.jsonl byte-stably for both 1-member and
20-member sets. status_override survives round-trip; INVALIDATED/DELETED
overrides land back as Status enum values.

Plausible wrong implementations:
- Stringify Status enum repr instead of .value.
- Write members as a single JSON array (loses the per-event JSONL parity
  the protocol mirrors from LedgerSession.export_jsonl).
- Silently accept duplicate member_id on read.
"""

from pathlib import Path

import pytest

from ledger_progress.core import Status
from ledger_progress.set_core import LedgerSet, LedgerSetMember
from ledger_progress.set_serialization import read_set_jsonl, write_set_jsonl


def test_roundtrip_singleton(tmp_path):
    s = LedgerSet("pilot_s_01", members=[LedgerSetMember("M1", "runs/swe_agent_pilot/swe_agent_pilot_s_01/ledger.jsonl")])
    path = tmp_path / "set.jsonl"

    write_set_jsonl(s, str(path))
    loaded = read_set_jsonl(str(path))

    assert loaded == s
    assert path.read_bytes() == _rewrite(loaded, tmp_path / "again.jsonl")


def test_roundtrip_20_members_with_overrides(tmp_path):
    members = [
        LedgerSetMember(f"M{i}", f"runs/p{i}/ledger.jsonl", weight=1.0 + i * 0.1)
        for i in range(1, 21)
    ]
    members[3].status_override = Status.INVALIDATED
    members[7].status_override = Status.DELETED
    s = LedgerSet("rollup", members=members)
    path = tmp_path / "set.jsonl"

    write_set_jsonl(s, str(path))
    loaded = read_set_jsonl(str(path))

    assert loaded == s
    assert loaded.members[3].status_override is Status.INVALIDATED
    assert loaded.members[7].status_override is Status.DELETED
    assert sum(1 for m in loaded.members if m.status_override is None) == 18


def test_status_override_serialized_as_stable_string_value(tmp_path):
    s = LedgerSet("x", members=[LedgerSetMember("M1", "a.jsonl", status_override=Status.INVALIDATED)])
    path = tmp_path / "set.jsonl"
    write_set_jsonl(s, str(path))

    text = path.read_text()

    assert '"status":"invalidated"' in text
    assert "Status." not in text


def test_duplicate_member_id_on_read_rejected(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        '{"type":"set_init","set_id":"x"}\n'
        '{"type":"add_member","member_id":"M1","ledger_ref":"a.jsonl","weight":1.0}\n'
        '{"type":"add_member","member_id":"M1","ledger_ref":"b.jsonl","weight":1.0}\n'
    )

    with pytest.raises(ValueError, match="duplicate member_id"):
        read_set_jsonl(str(path))


def test_missing_set_init_rejected(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text('{"type":"add_member","member_id":"M1","ledger_ref":"a.jsonl","weight":1.0}\n')

    with pytest.raises(ValueError, match="first line must be set_init"):
        read_set_jsonl(str(path))


def test_member_validation():
    with pytest.raises(ValueError, match="member_id is required"):
        LedgerSetMember("", "a.jsonl")
    with pytest.raises(ValueError, match="ledger_ref is required"):
        LedgerSetMember("M1", "")
    with pytest.raises(ValueError, match="weight must be positive"):
        LedgerSetMember("M1", "a.jsonl", weight=0.0)


def test_set_validation_rejects_duplicate_member_ids():
    with pytest.raises(ValueError, match="member_ids must be unique"):
        LedgerSet("x", members=[
            LedgerSetMember("M1", "a.jsonl"),
            LedgerSetMember("M1", "b.jsonl"),
        ])


def _rewrite(s, path: Path) -> bytes:
    write_set_jsonl(s, str(path))
    return path.read_bytes()
