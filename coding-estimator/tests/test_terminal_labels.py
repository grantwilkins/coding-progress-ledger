"""Terminal labels (Workstream E1).

Claim:
    `submit_without_validation_terminal(run)` mirrors upstream
    `state.submit_without_validation` evaluated at the run's terminal
    step. Specifically:
      - `validation_complete` is a STICKY accumulator: any
        UPDATE_STATUS event with status=='complete' on a VALIDATION
        subtask flips it to True forever, even if the subtask is
        later reopened or invalidated.
      - returns True iff there is an artifact-submit AND not (sticky
        validation_complete AND a validation subtask was ever added).
    `terminal_labels(run, label)` propagates FinalLabel fields verbatim
    (final_success, finish_step, finish_seconds, timeout) and tags a
    fifth target `y_submit_without_validation`.

Plausible wrong implementations:
    - inspect the FINAL ledger's validation status (s.status is
      Status.COMPLETE) instead of accumulating across events; this
      flips False on any "validation completed then reopened"
      run, which is the canonical golden fixture's exact shape.
    - require validation_complete in the open-window (t, terminal] only
    - require artifact submit AND validation NEVER attempted (instead
      of NEVER-completed) -> would mis-label runs where a validation
      subtask exists but never reaches complete.
    - misapply Status.INVALIDATED as Status.COMPLETE.
    - silently fabricate finish_seconds=0.0 instead of None when the
      label has finish_seconds=None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ledger_progress.serialization import event_from_dict, load_events_jsonl

from coding_estimator.ingest.labels import FinalLabel
from coding_estimator.ingest.run_record import RunRecord, load_run
from coding_estimator.labels.terminal import (
    submit_without_validation_terminal,
    terminal_labels,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_run"


def _run(events: list[dict[str, Any]], run_id: str = "synthetic") -> RunRecord:
    """Construct a RunRecord directly from synthetic event dicts. We use
    `event_from_dict` so the events go through the upstream serializer
    (and thus inherit upstream schema validation)."""
    objs = tuple(event_from_dict(e) for e in events)
    return RunRecord(
        run_id=run_id,
        source="tb_live",
        ledger_path=Path("/dev/null/synthetic_ledger.jsonl"),
        events=objs,
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id=run_id,
        task_family=None,
        arm=None,
        difficulty=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


def _validation_complete_then_reopen_with_submit() -> RunRecord:
    """Construct the regression scenario: validation completed, then
    reopened. Final ledger status of the VALIDATION subtask is
    IN_PROGRESS (not COMPLETE). An artifact-submit subtask is also
    completed.

    The non-sticky implementation would compute submit_without_validation
    = True (artifact submitted, final-status validation not COMPLETE).
    The correct sticky implementation must compute False (validation
    DID once complete, so the agent knew about validation before
    submitting).
    """
    return _run([
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "synthetic"}, "reason": None,
         "timestamp": "2026-05-04T00:00:00Z"},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "v1",
         "payload": {"description": "validate", "category": "validation",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:01Z"},
        {"step": 2, "event_type": "update_status", "subtask_id": "v1",
         "payload": {"status": "complete", "evidence": ["unit tests pass"]},
         "reason": None, "timestamp": "2026-05-04T00:00:02Z"},
        {"step": 3, "event_type": "reopen_subtask", "subtask_id": "v1",
         "payload": {}, "reason": "regression spotted",
         "timestamp": "2026-05-04T00:00:03Z"},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "a1",
         "payload": {"description": "submit final patch",
                     "category": "artifact", "weight": 1.0,
                     "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:04Z"},
        {"step": 5, "event_type": "update_status", "subtask_id": "a1",
         "payload": {"status": "complete", "evidence": ["patch submitted"]},
         "reason": None, "timestamp": "2026-05-04T00:00:05Z"},
    ])


def _validation_invalidated_then_submit() -> RunRecord:
    """Validation completed, then INVALIDATE_SUBTASK. Same regression
    contract: the sticky accumulator must remember the prior 'complete'
    event despite the later invalidation."""
    return _run([
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "synthetic"}, "reason": None,
         "timestamp": "2026-05-04T00:00:00Z"},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "v1",
         "payload": {"description": "validate", "category": "validation",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:01Z"},
        {"step": 2, "event_type": "update_status", "subtask_id": "v1",
         "payload": {"status": "complete", "evidence": ["green"]},
         "reason": None, "timestamp": "2026-05-04T00:00:02Z"},
        {"step": 3, "event_type": "invalidate_subtask", "subtask_id": "v1",
         "payload": {}, "reason": "harness wrong",
         "timestamp": "2026-05-04T00:00:03Z"},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "a1",
         "payload": {"description": "submit", "category": "artifact",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:04Z"},
        {"step": 5, "event_type": "update_status", "subtask_id": "a1",
         "payload": {"status": "complete", "evidence": ["filed"]},
         "reason": None, "timestamp": "2026-05-04T00:00:05Z"},
    ])


def _submit_without_any_validation_attempt() -> RunRecord:
    """Artifact-submit completed; NO validation subtask ever added.
    Both implementations agree: returns True (the canonical
    submit-without-validation case)."""
    return _run([
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "synthetic"}, "reason": None,
         "timestamp": "2026-05-04T00:00:00Z"},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "a1",
         "payload": {"description": "submit fix", "category": "artifact",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:01Z"},
        {"step": 2, "event_type": "update_status", "subtask_id": "a1",
         "payload": {"status": "complete", "evidence": ["submitted"]},
         "reason": None, "timestamp": "2026-05-04T00:00:02Z"},
    ])


def test_sticky_validation_complete_with_reopen_does_not_flip_label() -> None:
    """Regression: validation reaches COMPLETE then is reopened. The
    final ledger shows the validation subtask status != COMPLETE, so a
    final-ledger inspection would WRONGLY return True. The contract
    (matching upstream `state.validation_complete`) is sticky -> False.
    """
    run = _validation_complete_then_reopen_with_submit()
    assert submit_without_validation_terminal(run) is False


def test_sticky_validation_complete_with_invalidate_does_not_flip_label() -> None:
    """Same sticky contract, but with INVALIDATE_SUBTASK as the
    later event instead of REOPEN_SUBTASK. The final ledger's subtask
    has status=INVALIDATED; the sticky accumulator must still remember
    the prior 'complete' UPDATE_STATUS event."""
    run = _validation_invalidated_then_submit()
    assert submit_without_validation_terminal(run) is False


def test_artifact_submit_without_any_validation_returns_true() -> None:
    run = _submit_without_any_validation_attempt()
    assert submit_without_validation_terminal(run) is True


def test_no_artifact_submit_returns_false_regardless_of_validation_history() -> None:
    """Golden_run has validation completed AND invalidated, but NO
    artifact-submit subtask. Either implementation must return False
    because the artifact-submit precondition fails."""
    events = tuple(load_events_jsonl(str(FIXTURE_DIR / "ledger.jsonl")))
    run = RunRecord(
        run_id="golden",
        source="tb_live",
        ledger_path=FIXTURE_DIR / "ledger.jsonl",
        events=events,
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id="golden",
        task_family=None,
        arm=None,
        difficulty=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )
    assert submit_without_validation_terminal(run) is False


def test_terminal_labels_passes_through_finallabel_fields_verbatim() -> None:
    """terminal_labels must propagate FinalLabel fields without
    modification. A regression that defaults None -> 0.0 (silent
    fabrication) is the failure mode this guards against."""
    run = _submit_without_any_validation_attempt()
    label = FinalLabel(
        final_success=True,
        final_success_source="verifier_exit",
        finish_step=5,
        finish_seconds=None,  # synthetic source has no wallclock
        timeout=False,
        termination_reason=None,
    )
    out = terminal_labels(run, label)
    assert out.y_success_eventual is True
    assert out.y_finish_step == 5
    assert out.y_finish_seconds is None  # NOT 0.0 / NOT 0
    assert out.y_timeout is False
    # And the cross-cut: this run has artifact-submit but no validation,
    # so y_submit_without_validation must be True.
    assert out.y_submit_without_validation is True


def test_terminal_labels_parity_with_upstream_label_final_success() -> None:
    """For real canonical runs, `y_success_eventual` must equal the
    FinalLabel.final_success value (the contract says these labels are
    re-named pass-throughs of the upstream label_final_success
    columns)."""
    run = load_run("swe_agent_pilot", "swe_agent_pilot_s_06")
    out = terminal_labels(run)
    assert out.y_success_eventual is True  # pinned in test_label_loader.py
    assert out.y_finish_step == max(e.step for e in run.events)


def test_terminal_labels_finish_seconds_present_only_for_real_wallclock() -> None:
    """tb_live has real wallclock; swe_agent_pilot does not. The two
    must differ on finish_seconds presence — a failure here means a
    run-constant feature/label leakage path is silently fabricating."""
    tb = load_run("tb_live", "markdown-to-html-cli")
    swe = load_run("swe_agent_pilot", "swe_agent_pilot_s_06")
    out_tb = terminal_labels(tb)
    out_swe = terminal_labels(swe)
    assert out_tb.y_finish_seconds is not None and out_tb.y_finish_seconds > 0
    assert out_swe.y_finish_seconds is None
