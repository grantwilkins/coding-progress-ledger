from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import EventType, Ledger, LedgerEvent, replay


def event_to_dict(event: LedgerEvent) -> dict[str, Any]:
    return {
        "step": event.step,
        "event_type": event.event_type.value,
        "subtask_id": event.subtask_id,
        "payload": _jsonable(event.payload),
        "reason": event.reason,
    }


def event_from_dict(data: dict[str, Any]) -> LedgerEvent:
    return LedgerEvent(data["step"], EventType(data["event_type"]), data.get("subtask_id"), data["payload"], data.get("reason"))


def write_events_jsonl(events: list[LedgerEvent], path: str) -> None:
    Path(path).write_text("".join(json.dumps(event_to_dict(event), separators=(",", ":")) + "\n" for event in events))


def load_events_jsonl(path: str) -> list[LedgerEvent]:
    return [event_from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def to_jsonl(ledger: Ledger, path: str) -> None:
    write_events_jsonl(ledger.events, path)


def from_jsonl(path: str) -> Ledger:
    return replay(load_events_jsonl(path))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
