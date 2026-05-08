"""Snapshot of upstream Q1 horizon-label definitions.

Source: ../coding-progress-ledger/scripts/build_q_labels.py
Snapshot date: 2026-05-04
Snapshot SHA256 of upstream file at snapshot time:
  9a6f00185503e53d88640c6025de22cc2c9e5b604047447cdb37baaef14ff8fa

Reason: upstream `scripts/` is not packaged. Per AGENTS.md and
TASKS.md § E4, do NOT diverge from upstream without bumping the snapshot
date AND the SHA recorded above. The drift test in
tests/test_upstream_q_label_drift.py forces a deliberate re-snapshot.
"""

from __future__ import annotations

from ledger_progress.core import EventType, SubtaskCategory, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score

UPSTREAM_FILE_RELPATH = "scripts/build_q_labels.py"
SNAPSHOT_SHA256 = "9a6f00185503e53d88640c6025de22cc2c9e5b604047447cdb37baaef14ff8fa"

DISCOVERY_CATEGORIES = {SubtaskCategory.PRODUCT, SubtaskCategory.INVESTIGATION}


def _events_through_step(events, step):
    return [e for e in events if e.step <= step]


def _events_in_open_window(events, low_exclusive, high_inclusive):
    return [e for e in events if low_exclusive < e.step <= high_inclusive]


def _coding_progress(events_so_far):
    return (
        score(replay(events_so_far), CODING_CATEGORIES).progress
        if events_so_far
        else 0.0
    )


def _category_at(events_so_far, subtask_id):
    if not subtask_id:
        return None
    sub = replay(events_so_far).subtasks.get(subtask_id)
    return sub.category if sub else None


def _add_subtask_category(event):
    raw = event.payload.get("category", "product")
    return raw if isinstance(raw, SubtaskCategory) else SubtaskCategory(raw)


def _is_validation_transition(event, prefix_events):
    if event.event_type is not EventType.UPDATE_STATUS:
        return False
    if event.payload.get("status") not in {"complete", "blocked"}:
        return False
    return _category_at(prefix_events, event.subtask_id) is SubtaskCategory.VALIDATION


def _is_discovery_event(event, prefix_events):
    if event.event_type is EventType.ADD_SUBTASK:
        return _add_subtask_category(event) in DISCOVERY_CATEGORIES
    if event.event_type is EventType.REOPEN_SUBTASK:
        return _category_at(prefix_events, event.subtask_id) in DISCOVERY_CATEGORIES
    return False


def label_future_progress_drop(events, checkpoint_step, horizon, current_progress):
    prefix = _events_through_step(events, checkpoint_step)
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        prefix.append(e)
        if _coding_progress(prefix) < current_progress - 1e-9:
            return True
    return False


def label_validation_exposes_new_work(events, checkpoint_step, horizon):
    prefix = _events_through_step(events, checkpoint_step)
    saw_validation = False
    for e in _events_in_open_window(events, checkpoint_step, checkpoint_step + horizon):
        if not saw_validation and _is_validation_transition(e, prefix):
            saw_validation = True
        elif saw_validation and _is_discovery_event(e, prefix):
            return True
        prefix.append(e)
    return False
