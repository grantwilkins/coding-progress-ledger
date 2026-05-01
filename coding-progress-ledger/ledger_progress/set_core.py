from __future__ import annotations

from dataclasses import dataclass, field

from .core import Status


@dataclass
class LedgerSetMember:
    member_id: str
    ledger_ref: str
    weight: float = 1.0
    status_override: Status | None = None

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("member_id is required")
        if not self.ledger_ref:
            raise ValueError("ledger_ref is required")
        if self.weight <= 0:
            raise ValueError("member weight must be positive")
        self.status_override = _status_or_none(self.status_override)


@dataclass
class LedgerSet:
    set_id: str
    members: list[LedgerSetMember] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.set_id:
            raise ValueError("set_id is required")
        ids = [m.member_id for m in self.members]
        if len(set(ids)) != len(ids):
            raise ValueError("member_ids must be unique within a set")


def _status_or_none(value: str | Status | None) -> Status | None:
    if value is None:
        return None
    return value if isinstance(value, Status) else Status(value)
