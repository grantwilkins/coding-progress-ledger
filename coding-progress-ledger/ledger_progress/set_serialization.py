from __future__ import annotations

import json
from pathlib import Path

from .core import Status
from .set_core import LedgerSet, LedgerSetMember


SET_INIT = "set_init"
ADD_MEMBER = "add_member"
MARK_MEMBER = "mark_member"


def write_set_jsonl(ledger_set: LedgerSet, path: str) -> None:
    Path(path).write_text("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in _to_lines(ledger_set)))


def read_set_jsonl(path: str) -> LedgerSet:
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != SET_INIT:
        raise ValueError("first line must be set_init")
    ledger_set = LedgerSet(lines[0]["set_id"])
    by_id: dict[str, LedgerSetMember] = {}
    for line in lines[1:]:
        line_type = line.get("type")
        if line_type == ADD_MEMBER:
            if line["member_id"] in by_id:
                raise ValueError(f"duplicate member_id: {line['member_id']}")
            member = LedgerSetMember(
                member_id=line["member_id"],
                ledger_ref=line["ledger_ref"],
                weight=float(line.get("weight", 1.0)),
            )
            ledger_set.members.append(member)
            by_id[member.member_id] = member
        elif line_type == MARK_MEMBER:
            if line["member_id"] not in by_id:
                raise ValueError(f"unknown member_id: {line['member_id']}")
            by_id[line["member_id"]].status_override = Status(line["status"])
        else:
            raise ValueError(f"unknown set event type: {line_type}")
    return ledger_set


def _to_lines(ledger_set: LedgerSet) -> list[dict]:
    lines: list[dict] = [{"type": SET_INIT, "set_id": ledger_set.set_id}]
    for member in ledger_set.members:
        lines.append({
            "type": ADD_MEMBER,
            "member_id": member.member_id,
            "ledger_ref": member.ledger_ref,
            "weight": member.weight,
        })
        if member.status_override is not None:
            lines.append({
                "type": MARK_MEMBER,
                "member_id": member.member_id,
                "status": member.status_override.value,
            })
    return lines
