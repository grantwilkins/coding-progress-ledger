from .core import Ledger, ProgressObservation, Status


def score(ledger: Ledger) -> ProgressObservation:
    active_ids = {sid for sid, subtask in ledger.subtasks.items() if _active(subtask.status)}
    parents = {subtask.parent_id for subtask in ledger.subtasks.values() if subtask.parent_id in active_ids and _active(subtask.status)}
    leaves = [ledger.subtasks[sid] for sid in active_ids - parents]
    complete_weight = sum(subtask.weight for subtask in leaves if subtask.status is Status.COMPLETE)
    active_weight = sum(subtask.weight for subtask in leaves)
    return ProgressObservation(
        step=ledger.events[-1].step if ledger.events else 0,
        complete_weight=complete_weight,
        active_weight=active_weight,
        progress=complete_weight / active_weight if active_weight else 0.0,
        complete_leaf_count=sum(1 for subtask in leaves if subtask.status is Status.COMPLETE),
        active_leaf_count=len(leaves),
    )


def _active(status: Status) -> bool:
    return status not in {Status.INVALIDATED, Status.DELETED}
