from __future__ import annotations

from collections.abc import Iterable

from .core import Ledger, Status, Subtask, SubtaskCategory


CODING_CATEGORIES = (
    SubtaskCategory.PRODUCT,
    SubtaskCategory.VALIDATION,
    SubtaskCategory.INVESTIGATION,
)


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
