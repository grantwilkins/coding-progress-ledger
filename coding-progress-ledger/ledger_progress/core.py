from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    INVALIDATED = "invalidated"
    DELETED = "deleted"


class EventType(Enum):
    INIT = "init"
    ADD_SUBTASK = "add_subtask"
    UPDATE_STATUS = "update_status"
    ADD_EVIDENCE = "add_evidence"
    SPLIT_SUBTASK = "split_subtask"
    REOPEN_SUBTASK = "reopen_subtask"
    INVALIDATE_SUBTASK = "invalidate_subtask"
    DELETE_SUBTASK = "delete_subtask"


@dataclass
class Subtask:
    id: str
    description: str
    status: Status
    evidence: list[str] = field(default_factory=list)
    weight: float = 1.0
    parent_id: str | None = None
    created_at_step: int = 0
    updated_at_step: int = 0

    def __post_init__(self) -> None:
        self.status = _status(self.status)
        if not self.id or not self.description:
            raise ValueError("subtask id and description are required")
        if self.weight <= 0:
            raise ValueError("subtask weight must be positive")
        _strings(self.evidence, "evidence")


@dataclass
class LedgerEvent:
    step: int
    event_type: EventType
    subtask_id: str | None
    payload: dict[str, Any]
    reason: str | None = None

    def __post_init__(self) -> None:
        self.event_type = EventType(self.event_type)
        if self.step < 0:
            raise ValueError("event step must be non-negative")
        if not isinstance(self.payload, dict):
            raise ValueError("event payload must be a dict")


@dataclass
class Ledger:
    root_task: str
    subtasks: dict[str, Subtask] = field(default_factory=dict)
    events: list[LedgerEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.root_task:
            raise ValueError("root_task is required")


@dataclass
class ProgressObservation:
    step: int
    complete_weight: float
    active_weight: float
    progress: float
    complete_leaf_count: int
    active_leaf_count: int


def new_ledger(root_task: str) -> Ledger:
    return Ledger(root_task, events=[LedgerEvent(0, EventType.INIT, None, {"root_task": root_task})])


def apply_event(ledger: Ledger, event: LedgerEvent) -> Ledger:
    handlers = {
        EventType.ADD_SUBTASK: _add_subtask,
        EventType.UPDATE_STATUS: _update_status,
        EventType.ADD_EVIDENCE: _add_evidence,
        EventType.SPLIT_SUBTASK: _split_subtask,
        EventType.REOPEN_SUBTASK: lambda l, e: _set_status(l, e, Status.IN_PROGRESS),
        EventType.INVALIDATE_SUBTASK: lambda l, e: _set_status(l, e, Status.INVALIDATED),
        EventType.DELETE_SUBTASK: lambda l, e: _set_status(l, e, Status.DELETED),
    }
    if event.event_type not in handlers:
        raise ValueError("init events are only valid as the first replay event")
    handlers[event.event_type](ledger, event)
    ledger.events.append(event)
    return ledger


def replay(events: list[LedgerEvent]) -> Ledger:
    if not events or EventType(events[0].event_type) is not EventType.INIT:
        raise ValueError("first event must be init")
    root_task = events[0].payload.get("root_task")
    ledger = Ledger(root_task, events=[events[0]])
    for event in events[1:]:
        apply_event(ledger, event)
    return ledger


def _add_subtask(ledger: Ledger, event: LedgerEvent) -> None:
    subtask_id = _subtask_id(event)
    if subtask_id in ledger.subtasks:
        raise ValueError(f"duplicate subtask id: {subtask_id}")
    parent_id = event.payload.get("parent_id")
    if parent_id is not None and parent_id not in ledger.subtasks:
        raise ValueError(f"unknown parent_id: {parent_id}")
    ledger.subtasks[subtask_id] = Subtask(
        id=subtask_id,
        description=_required(event, "description"),
        status=_status(event.payload.get("status", Status.NOT_STARTED)),
        weight=float(event.payload.get("weight", 1.0)),
        parent_id=parent_id,
        created_at_step=event.step,
        updated_at_step=event.step,
    )


def _update_status(ledger: Ledger, event: LedgerEvent) -> None:
    subtask = _subtask(ledger, event)
    status = _status(_required(event, "status"))
    evidence = event.payload.get("evidence")
    evidence = [] if evidence is None else _strings(evidence, "evidence")
    if status is Status.COMPLETE and not subtask.evidence and not evidence:
        raise ValueError("complete subtasks require evidence")
    subtask.evidence.extend(evidence)
    subtask.status = status
    subtask.updated_at_step = event.step


def _add_evidence(ledger: Ledger, event: LedgerEvent) -> None:
    evidence = _strings(_required(event, "evidence"), "evidence")
    if not evidence:
        raise ValueError("evidence must be non-empty")
    subtask = _subtask(ledger, event)
    subtask.evidence.extend(evidence)
    subtask.updated_at_step = event.step


def _split_subtask(ledger: Ledger, event: LedgerEvent) -> None:
    parent_id = _subtask(ledger, event).id
    children = _required(event, "children")
    if not isinstance(children, list) or not children:
        raise ValueError("children must be a non-empty list")
    ids = [child.get("id") for child in children if isinstance(child, dict)]
    if len(ids) != len(children) or len(set(ids)) != len(ids) or any(i in ledger.subtasks for i in ids):
        raise ValueError("child ids must be unique and new")
    subtasks = [
        Subtask(
            id=child["id"],
            description=child["description"],
            status=_status(child.get("status", Status.NOT_STARTED)),
            weight=float(child.get("weight", 1.0)),
            parent_id=parent_id,
            created_at_step=event.step,
            updated_at_step=event.step,
        )
        for child in children
    ]
    ledger.subtasks.update({subtask.id: subtask for subtask in subtasks})


def _set_status(ledger: Ledger, event: LedgerEvent, status: Status) -> None:
    subtask = _subtask(ledger, event)
    subtask.status = status
    subtask.updated_at_step = event.step


def _status(value: str | Status) -> Status:
    return value if isinstance(value, Status) else Status(value)


def _subtask_id(event: LedgerEvent) -> str:
    if not event.subtask_id:
        raise ValueError(f"{event.event_type.value} requires subtask_id")
    return event.subtask_id


def _subtask(ledger: Ledger, event: LedgerEvent) -> Subtask:
    subtask_id = _subtask_id(event)
    if subtask_id not in ledger.subtasks:
        raise ValueError(f"unknown subtask id: {subtask_id}")
    return ledger.subtasks[subtask_id]


def _required(event: LedgerEvent, key: str) -> Any:
    if key not in event.payload:
        raise ValueError(f"{event.event_type.value} requires payload.{key}")
    return event.payload[key]


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return value
