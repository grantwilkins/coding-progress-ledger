from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    INVESTIGATION = "INVESTIGATION"
    PRODUCT = "PRODUCT"
    VALIDATION = "VALIDATION"
    ENVIRONMENT = "ENVIRONMENT"
    ARTIFACT = "ARTIFACT"


@dataclass(frozen=True)
class Turn:
    step: int
    kind: str
    tool: str | None = None
    command: str | None = None
    response: str | None = None
    arguments: dict[str, Any] | None = None
    source: str | None = None
    instance_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        obj = asdict(self)
        return {key: value for key, value in obj.items() if value not in (None, {}, [])}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "Turn":
        if not isinstance(obj, dict):
            raise TypeError("turn JSON row must be an object")
        return cls(
            step=int(obj["step"]),
            kind=str(obj["kind"]),
            tool=obj.get("tool"),
            command=obj.get("command"),
            response=obj.get("response"),
            arguments=obj.get("arguments"),
            source=obj.get("source"),
            instance_id=obj.get("instance_id"),
            metadata=dict(obj.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Row:
    step: int
    total: int
    done: int
    current_category: str
    current_unit_age: int
    had_stuck_episode: bool
    kind: str
    tool: str

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Summary:
    instance_id: str
    final_total: int
    final_done: int
    had_stuck_episode: bool
    exit_status: str

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)
