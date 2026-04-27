from collections.abc import Iterable

from .core import Ledger, ProgressObservation, Status, SubtaskCategory


def score(ledger: Ledger, categories: Iterable[SubtaskCategory | str] | None = None) -> ProgressObservation:
    selected_categories = _selected_categories(categories)
    active_ids = {sid for sid, subtask in ledger.subtasks.items() if _active(subtask.status)}
    parents = {subtask.parent_id for subtask in ledger.subtasks.values() if subtask.parent_id in active_ids and _active(subtask.status)}
    leaves = [ledger.subtasks[sid] for sid in active_ids - parents if ledger.subtasks[sid].category in selected_categories]
    complete_weight = sum(subtask.weight for subtask in leaves if subtask.status is Status.COMPLETE)
    active_weight = sum(subtask.weight for subtask in leaves)
    return ProgressObservation(
        step=ledger.events[-1].step if ledger.events else 0,
        categories_included=tuple(category for category in SubtaskCategory if category in selected_categories),
        complete_weight=complete_weight,
        active_weight=active_weight,
        progress=complete_weight / active_weight if active_weight else 0.0,
        complete_leaf_count=sum(1 for subtask in leaves if subtask.status is Status.COMPLETE),
        active_leaf_count=len(leaves),
    )


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
