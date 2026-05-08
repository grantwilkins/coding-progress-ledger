"""Snapshot of upstream observation-shape labelling.

Source: ../coding-progress-ledger/scripts/label_observation_shapes.py
Snapshot date: 2026-05-04
Snapshot SHA256 of upstream file at snapshot time:
  021b089f10f3ada39a5118da264c3ed311409569f59c5dda3302cbe761f99aae

Shape labels are slice tags only (NOT prediction targets in v0). The
copy here mirrors `label_run` and its helpers byte-for-byte to keep
parity with upstream `*_shape_labels.csv` outputs. Drift test in
tests/test_upstream_shapes_drift.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ledger_progress.core import EventType, Status, SubtaskCategory, replay
from ledger_progress.run_manager import resolve_final_success
from ledger_progress.serialization import event_from_dict

UPSTREAM_FILE_RELPATH = "scripts/label_observation_shapes.py"
SNAPSHOT_SHA256 = "021b089f10f3ada39a5118da264c3ed311409569f59c5dda3302cbe761f99aae"

HIGH_PROGRESS_THRESHOLD = 0.70

SHAPE_TAGS = (
    "high_progress_failure",
    "low_progress_success",
    "stuck_loop",
    "submit_without_validation",
    "no_validation_frontier",
    "validation_induced_reopen",
    "scope_discovery_after_high_progress",
    "hidden_work_gap",
    "nonmonotone_recovery",
)

HIDDEN_WORK_PHRASES = (
    "hidden-work",
    "hidden work",
    "DID NOT trigger",
    "did not trigger",
    "uninformative",
    "repro was insufficient",
    "insufficient",
)

VALIDATION_REOPEN_PHRASES = (
    "repro",
    "traceback",
    "pytest",
    "re-run",
    "still emits",
    "still raises",
    "rerun",
)


@dataclass
class RunLabels:
    run_id: str
    final_coding_progress: float
    final_success: bool | None
    final_success_source: str
    tags: set[str] = field(default_factory=set)
    clean_success: bool = False
    notes: list[str] = field(default_factory=list)


def _load_events(run_dir: Path):
    events = []
    with (run_dir / "ledger.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(event_from_dict(json.loads(line)))
    return events


def _evidence_matches(ledger, phrases: tuple[str, ...]) -> bool:
    needles = tuple(p.lower() for p in phrases)
    for event in ledger.events:
        if event.event_type is not EventType.UPDATE_STATUS:
            continue
        for ev in event.payload.get("evidence", []) or []:
            if any(n in ev.lower() for n in needles):
                return True
    return False


def _has_blocked_subtask(ledger) -> bool:
    return any(s.status is Status.BLOCKED for s in ledger.subtasks.values())


def _has_validation_subtask(ledger) -> bool:
    return any(
        s.category is SubtaskCategory.VALIDATION for s in ledger.subtasks.values()
    )


def _has_completed_validation(ledger) -> bool:
    return any(
        s.category is SubtaskCategory.VALIDATION and s.status is Status.COMPLETE
        for s in ledger.subtasks.values()
    )


def _has_completed_artifact_submit(ledger) -> bool:
    for sub in ledger.subtasks.values():
        if sub.category is not SubtaskCategory.ARTIFACT or sub.status is not Status.COMPLETE:
            continue
        desc = (sub.description or "").lower()
        if "submit" in desc:
            return True
    for event in ledger.events:
        if event.event_type is not EventType.UPDATE_STATUS:
            continue
        sid = event.subtask_id
        if sid is None:
            continue
        sub = ledger.subtasks.get(sid)
        if sub is None or sub.category is not SubtaskCategory.ARTIFACT:
            continue
        if event.payload.get("status") != "complete":
            continue
        for ev in event.payload.get("evidence", []) or []:
            if "submit" in ev.lower():
                return True
    return False


def _validation_induced_reopen(ledger) -> bool:
    for event in ledger.events:
        if event.event_type is not EventType.REOPEN_SUBTASK:
            continue
        reason = (event.reason or "") + " " + str(event.payload.get("reason") or "")
        if any(phrase in reason.lower() for phrase in VALIDATION_REOPEN_PHRASES):
            return True
    return False


def _scope_discovery_after_high_progress(events) -> bool:
    seen_reopen = False
    for event in events:
        if event.event_type is EventType.REOPEN_SUBTASK:
            seen_reopen = True
            continue
        if not seen_reopen:
            continue
        if event.event_type is EventType.ADD_SUBTASK:
            cat = event.payload.get("category")
            if cat in {"product", "investigation"}:
                return True
    return False


def _nonmonotone_recovery(ledger, summary: dict, success: bool | None) -> bool:
    if not summary.get("nonmonotonic_coding"):
        return False
    if not any(e.event_type is EventType.REOPEN_SUBTASK for e in ledger.events):
        return False
    if success is not True:
        return False
    return summary.get("final_coding_progress", 0.0) >= 1.0 - 1e-9


def label_run(run_dir: Path) -> RunLabels:
    events = _load_events(run_dir)
    ledger = replay(events)
    summary = json.loads((run_dir / "summary_by_category.json").read_text())
    success, source = resolve_final_success(run_dir, summary)
    coding = float(summary.get("final_coding_progress", 0.0))

    out = RunLabels(
        run_id=run_dir.name,
        final_coding_progress=coding,
        final_success=success,
        final_success_source=source,
    )
    if success is False and coding >= HIGH_PROGRESS_THRESHOLD:
        out.tags.add("high_progress_failure")
    if success is True and coding < HIGH_PROGRESS_THRESHOLD:
        out.tags.add("low_progress_success")
    if _has_blocked_subtask(ledger):
        out.tags.add("stuck_loop")
    if not _has_validation_subtask(ledger):
        out.tags.add("no_validation_frontier")
    if _has_completed_artifact_submit(ledger) and not _has_completed_validation(ledger):
        out.tags.add("submit_without_validation")
    if _validation_induced_reopen(ledger):
        out.tags.add("validation_induced_reopen")
    if _scope_discovery_after_high_progress(events):
        out.tags.add("scope_discovery_after_high_progress")
    if _evidence_matches(ledger, HIDDEN_WORK_PHRASES):
        out.tags.add("hidden_work_gap")
    if _nonmonotone_recovery(ledger, summary, success):
        out.tags.add("nonmonotone_recovery")

    out.clean_success = (
        success is True
        and "low_progress_success" not in out.tags
        and "submit_without_validation" not in out.tags
        and "no_validation_frontier" not in out.tags
        and "high_progress_failure" not in out.tags
    )
    return out
