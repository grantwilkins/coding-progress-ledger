from __future__ import annotations

from collections.abc import Iterable

from .core import EventType, Ledger, LedgerEvent, Status, Subtask, SubtaskCategory


CODING_CATEGORIES = (
    SubtaskCategory.PRODUCT,
    SubtaskCategory.VALIDATION,
    SubtaskCategory.INVESTIGATION,
)


def current_step(ledger: Ledger) -> int:
    return max((event.step for event in ledger.events), default=0)


def active_blocked_leaves(ledger: Ledger) -> list[Subtask]:
    return [s for s in active_incomplete_leaves(ledger) if s.status is Status.BLOCKED]


def reopens_since(ledger: Ledger, step: int) -> list[LedgerEvent]:
    return [e for e in ledger.events if e.step > step and e.event_type is EventType.REOPEN_SUBTASK]


def newly_discovered_since(ledger: Ledger, step: int) -> list[Subtask]:
    return [
        ledger.subtasks[sid]
        for sid, subtask in ledger.subtasks.items()
        if subtask.created_at_step > step
    ]


def last_validation_event(ledger: Ledger) -> LedgerEvent | None:
    val_ids = {sid for sid, s in ledger.subtasks.items() if s.category is SubtaskCategory.VALIDATION}
    if not val_ids:
        return None
    candidates = [
        e for e in ledger.events
        if e.subtask_id in val_ids and e.event_type in {
            EventType.UPDATE_STATUS, EventType.ADD_SUBTASK, EventType.ADD_EVIDENCE
        }
    ]
    return candidates[-1] if candidates else None


def stalled_for(ledger: Ledger, status: Status = Status.BLOCKED) -> int:
    blocked_ids = {sid for sid, s in ledger.subtasks.items() if s.status is status}
    if not blocked_ids:
        return 0
    earliest = min(ledger.subtasks[sid].updated_at_step for sid in blocked_ids)
    return current_step(ledger) - earliest


def active_incomplete_leaves(
    ledger: Ledger,
    categories: Iterable[SubtaskCategory | str] | None = None,
) -> list[Subtask]:
    selected_categories = _selected_categories(categories)
    active_ids = {sid for sid, subtask in ledger.subtasks.items() if _active(subtask.status)}
    parents = {
        subtask.parent_id
        for subtask in ledger.subtasks.values()
        if subtask.parent_id in active_ids and _active(subtask.status)
    }
    leaves = [
        ledger.subtasks[sid]
        for sid in active_ids - parents
        if ledger.subtasks[sid].category in selected_categories
        and ledger.subtasks[sid].status is not Status.COMPLETE
    ]
    return sorted(leaves, key=lambda subtask: (subtask.created_at_step, subtask.id))


def active_incomplete_coding_leaves(ledger: Ledger) -> list[Subtask]:
    return active_incomplete_leaves(ledger, CODING_CATEGORIES)


def _active(status: Status) -> bool:
    return status not in {Status.INVALIDATED, Status.DELETED}


def _selected_categories(categories: Iterable[SubtaskCategory | str] | None) -> set[SubtaskCategory]:
    if categories is None:
        return set(SubtaskCategory)
    if isinstance(categories, (str, SubtaskCategory)):
        categories = [categories]
    return {
        category if isinstance(category, SubtaskCategory) else SubtaskCategory(category)
        for category in categories
    }
