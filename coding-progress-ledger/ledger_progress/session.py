from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from .core import EventType, Ledger, LedgerEvent, Status, SubtaskCategory, apply_event, new_ledger, replay
from .scoring import score as score_ledger
from .serialization import to_jsonl


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LedgerSession:
    def __init__(self, root_task: str | Ledger, clock: Callable[[], str | None] | None = None):
        self.ledger = root_task if isinstance(root_task, Ledger) else new_ledger(root_task)
        self._clock = clock if clock is not None else _utc_now_iso

    def add(
        self,
        description: str,
        step: int,
        parent_id: str | None = None,
        weight: float = 1.0,
        reason: str | None = None,
        category: SubtaskCategory | str = SubtaskCategory.PRODUCT,
    ) -> str:
        subtask_id = self._next_id()
        category = _category(category)
        payload = {
            "description": description,
            "parent_id": parent_id,
            "weight": weight,
            "category": category,
        }
        self._apply(step, EventType.ADD_SUBTASK, subtask_id, payload, reason)
        return subtask_id

    def complete(self, subtask_id: str, evidence: str | list[str], step: int, reason: str | None = None) -> Ledger:
        return self._status(subtask_id, Status.COMPLETE, step, evidence, reason)

    def start(self, subtask_id: str, step: int, evidence: str | list[str] | None = None, reason: str | None = None) -> Ledger:
        return self._status(subtask_id, Status.IN_PROGRESS, step, evidence, reason)

    def block(self, subtask_id: str, step: int, reason: str, evidence: str | list[str] | None = None) -> Ledger:
        return self._status(subtask_id, Status.BLOCKED, step, evidence, reason)

    def reopen(self, subtask_id: str, step: int, reason: str) -> Ledger:
        return self._apply(step, EventType.REOPEN_SUBTASK, subtask_id, {"reason": reason}, reason)

    def invalidate(self, subtask_id: str, step: int, reason: str) -> Ledger:
        return self._apply(step, EventType.INVALIDATE_SUBTASK, subtask_id, {"reason": reason}, reason)

    def split(
        self,
        subtask_id: str,
        child_descriptions: list[str],
        step: int,
        reason: str,
        categories: list[SubtaskCategory | str | None] | None = None,
        category: SubtaskCategory | str | None = None,
    ) -> list[str]:
        if category is not None and categories is not None:
            raise ValueError("provide either category or categories, not both")
        if category is not None:
            categories = [category] * len(child_descriptions)
        if categories is not None and len(categories) != len(child_descriptions):
            raise ValueError("categories must match child_descriptions length")
        child_ids = self._child_ids(subtask_id, len(child_descriptions))
        children = []
        for sid, desc, category in zip(child_ids, child_descriptions, categories or [None] * len(child_descriptions)):
            child = {"id": sid, "description": desc}
            if category is not None:
                child["category"] = _category(category)
            children.append(child)
        self._apply(step, EventType.SPLIT_SUBTASK, subtask_id, {
            "children": children,
        }, reason)
        return child_ids

    def score(self, step: int | None = None, categories: Iterable[SubtaskCategory | str] | None = None):
        if step is None:
            return score_ledger(self.ledger, categories=categories)
        return score_ledger(replay([event for event in self.ledger.events if event.step <= step]), categories=categories)

    def export_jsonl(self, path: str) -> None:
        to_jsonl(self.ledger, path)

    def export_curve_csv(self, path: str) -> None:
        rows = [("step", "complete_weight", "active_weight", "progress", "complete_leaf_count", "active_leaf_count")]
        for step in sorted({event.step for event in self.ledger.events}):
            obs = self.score(step)
            rows.append((obs.step, obs.complete_weight, obs.active_weight, obs.progress, obs.complete_leaf_count, obs.active_leaf_count))
        with Path(path).open("w", newline="") as file:
            csv.writer(file).writerows(rows)

    def _status(self, subtask_id: str, status: Status, step: int, evidence: str | list[str] | None, reason: str | None) -> Ledger:
        payload = {"status": status}
        if evidence is not None:
            payload["evidence"] = [evidence] if isinstance(evidence, str) else evidence
        return self._apply(step, EventType.UPDATE_STATUS, subtask_id, payload, reason)

    def _apply(self, step: int, event_type: EventType, subtask_id: str, payload: dict, reason: str | None) -> Ledger:
        self.ledger = apply_event(self.ledger, LedgerEvent(step, event_type, subtask_id, payload, reason, self._clock()))
        return self.ledger

    def _next_id(self) -> str:
        i = len(self.ledger.subtasks) + 1
        while f"S{i}" in self.ledger.subtasks:
            i += 1
        return f"S{i}"

    def _child_ids(self, parent_id: str, count: int) -> list[str]:
        ids, i = [], 1
        while len(ids) < count:
            subtask_id = f"{parent_id}.{i}"
            if subtask_id not in self.ledger.subtasks:
                ids.append(subtask_id)
            i += 1
        return ids


LedgerBuilder = LedgerSession


def _category(value: SubtaskCategory | str) -> SubtaskCategory:
    return value if isinstance(value, SubtaskCategory) else SubtaskCategory(value)
