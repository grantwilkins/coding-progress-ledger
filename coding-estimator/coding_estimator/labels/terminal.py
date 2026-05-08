"""Terminal labels (run-constant; replicated to every checkpoint of a run).

Targets shipped here:
  y_success_eventual            — final_success bool (FinalLabel)
  y_finish_step                 — last event step
  y_finish_seconds              — wallclock duration if real-wallclock else None
  y_timeout                     — FinalLabel.timeout
  y_submit_without_validation   — terminal-state replay matches upstream
                                   `state.submit_without_validation`
                                   (build_estimator_checkpoints.py:207).

`y_submit_without_validation` is run-constant by definition: any
non-trivial AUROC at non-terminal t is a data property, not skill
(see registry.V0_TARGETS docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

from ledger_progress.core import EventType, Status, SubtaskCategory, replay

from coding_estimator.ingest.labels import FinalLabel, load_final_label
from coding_estimator.ingest.run_record import RunRecord


@dataclass(frozen=True)
class TerminalLabels:
    y_success_eventual: bool
    y_finish_step: int | None
    y_finish_seconds: float | None
    y_timeout: bool
    y_submit_without_validation: bool


def _has_artifact_submit(ledger) -> bool:
    return any(
        s.category is SubtaskCategory.ARTIFACT
        and s.status is Status.COMPLETE
        and (
            "submit" in (s.description or "").lower()
            or any("submit" in ev.lower() for ev in s.evidence)
        )
        for s in ledger.subtasks.values()
    )


def _has_validation_subtask(ledger) -> bool:
    return any(
        s.category is SubtaskCategory.VALIDATION for s in ledger.subtasks.values()
    )


def _validation_ever_completed(events, ledger) -> bool:
    """Sticky `validation_complete` accumulator. Mirrors upstream
    build_estimator_checkpoints._update_state: once any UPDATE_STATUS
    event sets a VALIDATION subtask to `complete`, the flag stays True
    even if the subtask is later reopened or invalidated. Inspecting
    the final ledger only would miss reopened-then-still-not-completed
    validations and silently flip the label."""
    for e in events:
        if e.event_type is not EventType.UPDATE_STATUS:
            continue
        if e.payload.get("status") != "complete":
            continue
        sub = ledger.subtasks.get(e.subtask_id)
        if sub is not None and sub.category is SubtaskCategory.VALIDATION:
            return True
    return False


def submit_without_validation_terminal(run: RunRecord) -> bool:
    """`state.submit_without_validation` evaluated at the terminal step.

    Mirrors upstream build_estimator_checkpoints._update_state line 207:
      submit_without_validation = has_artifact_submit
                                  and not (validation_complete and has_val_subtask)
    where `validation_complete` is the *sticky* accumulator (see
    `_validation_ever_completed`).
    """
    events = list(run.events)
    ledger = replay(events)
    val_complete = _validation_ever_completed(events, ledger)
    val_present = _has_validation_subtask(ledger)
    return _has_artifact_submit(ledger) and not (val_complete and val_present)


def terminal_labels(run: RunRecord, label: FinalLabel | None = None) -> TerminalLabels:
    if label is None:
        label = load_final_label(run)
    return TerminalLabels(
        y_success_eventual=label.final_success,
        y_finish_step=label.finish_step,
        y_finish_seconds=label.finish_seconds,
        y_timeout=label.timeout,
        y_submit_without_validation=submit_without_validation_terminal(run),
    )
