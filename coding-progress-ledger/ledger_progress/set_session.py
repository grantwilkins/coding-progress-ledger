from __future__ import annotations

from pathlib import Path

from .core import Status
from .scoring import score_set
from .set_core import LedgerSet, LedgerSetMember
from .set_serialization import write_set_jsonl


class LedgerSetSession:
    def __init__(self, set_id: str):
        self.set = LedgerSet(set_id)

    def add_member(self, ledger_ref: str, weight: float = 1.0, member_id: str | None = None) -> str:
        if member_id is None:
            member_id = f"M{len(self.set.members) + 1}"
        if any(m.member_id == member_id for m in self.set.members):
            raise ValueError(f"duplicate member_id: {member_id}")
        self.set.members.append(LedgerSetMember(member_id=member_id, ledger_ref=ledger_ref, weight=weight))
        return member_id

    def mark_member(self, member_id: str, status: Status | str) -> None:
        for member in self.set.members:
            if member.member_id == member_id:
                member.status_override = status if isinstance(status, Status) else Status(status)
                return
        raise ValueError(f"unknown member_id: {member_id}")

    def score(self, base_dir: str | Path | None = None) -> float:
        return score_set(self.set, base_dir=base_dir)

    def export_jsonl(self, path: str) -> None:
        write_set_jsonl(self.set, path)
