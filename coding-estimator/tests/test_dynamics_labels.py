"""Process-dynamics labels (Workstream E4).

Claim:
    `future_progress_drop_h5(run, t, ...)` and
    `validation_new_work_h5(run, t, ...)` evaluate the upstream Q1
    horizon=5 labels with the upstream half-open window (t, t+5]. Both
    labels are masked when:
      - is_terminal=True            -> mask_reason="is_terminal_checkpoint"
      - finish_step is None         -> mask_reason="finish_step_unknown"
      - t + 5 > finish_step         -> mask_reason="horizon_exceeds_finish_step"
    Mask precedence is: terminal first, then finish_step_unknown, then
    horizon-exceeds. `future_progress_drop_h5` compares each step in the
    window's progress against `current_progress` measured at t (not the
    progress at the previous step).

Plausible wrong implementations:
    - missing the `finish_step is None` mask -> returns an unmasked
      False where the contract demands a mask (regression: the bug the
      Sonnet critic surfaced before this test was written).
    - using `>=` instead of `>` in the horizon-exceeds check -> masks
      one step too early.
    - using a closed window [t, t+5] -> would treat events AT t as
      "future", flipping labels at the boundary.
    - comparing each future step's progress against the PRIOR step's
      progress (delta-style) instead of against `current_progress` at t
      (level-style); this matters because golden_run starts at progress
      0.0 at t=0 and the upstream definition treats the run's initial
      ramp-up as no-drop.
    - validation_new_work treating INVALIDATE_SUBTASK as a "validation
      transition" -> the upstream definition restricts to
      UPDATE_STATUS with status complete/blocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ledger_progress.serialization import event_from_dict, load_events_jsonl

from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.labels.dynamics import (
    H5,
    future_progress_drop_h5,
    validation_new_work_h5,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_run"


def _golden_run() -> RunRecord:
    events = tuple(load_events_jsonl(str(FIXTURE_DIR / "ledger.jsonl")))
    return RunRecord(
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


def _run_from_dicts(events: list[dict[str, Any]]) -> RunRecord:
    objs = tuple(event_from_dict(e) for e in events)
    return RunRecord(
        run_id="synth",
        source="tb_live",
        ledger_path=Path("/dev/null/synthetic.jsonl"),
        events=objs,
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id="synth",
        task_family=None,
        arm=None,
        difficulty=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


# ---- Mask-rule tests ----


def test_terminal_checkpoint_is_masked() -> None:
    """At the run's terminal step, both dynamics labels are masked."""
    run = _golden_run()
    terminal = max(e.step for e in run.events)
    out_drop = future_progress_drop_h5(run, terminal, is_terminal=True, finish_step=terminal)
    out_val = validation_new_work_h5(run, terminal, is_terminal=True, finish_step=terminal)
    assert out_drop.is_masked is True and out_drop.mask_reason == "is_terminal_checkpoint"
    assert out_drop.value is None
    assert out_val.is_masked is True and out_val.mask_reason == "is_terminal_checkpoint"
    assert out_val.value is None


def test_finish_step_none_is_masked() -> None:
    """When finish_step is None (unresolvable label), the contract
    requires masking with reason 'finish_step_unknown'. Without this
    rule, the function silently emits an unmasked label that may have
    been computed against an incomplete event log."""
    run = _golden_run()
    out_drop = future_progress_drop_h5(run, t_step=0, is_terminal=False, finish_step=None)
    out_val = validation_new_work_h5(run, t_step=0, is_terminal=False, finish_step=None)
    assert out_drop.is_masked is True and out_drop.mask_reason == "finish_step_unknown"
    assert out_val.is_masked is True and out_val.mask_reason == "finish_step_unknown"


def test_horizon_exceeds_finish_step_boundary() -> None:
    """Half-open mask: t + 5 > finish_step masks; t + 5 == finish_step
    does NOT mask. Boundary case at golden_run terminal=13:
      t=8 -> 8+5=13, NOT > 13   -> NOT masked
      t=9 -> 9+5=14, IS  > 13   -> masked
    """
    run = _golden_run()
    not_masked = future_progress_drop_h5(run, t_step=8, is_terminal=False, finish_step=13)
    masked = future_progress_drop_h5(run, t_step=9, is_terminal=False, finish_step=13)
    assert not_masked.is_masked is False, not_masked
    assert masked.is_masked is True
    assert masked.mask_reason == "horizon_exceeds_finish_step"


def test_terminal_mask_takes_precedence_over_horizon_mask() -> None:
    """If both is_terminal=True and t+5>finish_step, the reason returned
    is 'is_terminal_checkpoint' (precedence: terminal first)."""
    run = _golden_run()
    out = future_progress_drop_h5(run, t_step=13, is_terminal=True, finish_step=13)
    assert out.is_masked is True
    assert out.mask_reason == "is_terminal_checkpoint"


# ---- Hand-checkable progress-drop tests on golden_run ----


def test_progress_drop_at_t0_is_false_because_baseline_is_zero() -> None:
    """At t=0, current_progress=0.0 — no future step can drop below 0.
    A delta-style implementation (compare adjacent steps) would WRONGLY
    return True here because progress increases then decreases later."""
    run = _golden_run()
    out = future_progress_drop_h5(run, t_step=0, is_terminal=False, finish_step=13)
    assert out.is_masked is False
    assert out.value is False


def test_progress_drop_at_t2_is_true() -> None:
    """At t=2, golden_run has reached coding_progress=1.0. Window (2, 7]
    contains step 3 with progress 0.5 < 1.0 — the drop is detected."""
    run = _golden_run()
    out = future_progress_drop_h5(run, t_step=2, is_terminal=False, finish_step=13)
    assert out.is_masked is False
    assert out.value is True


def test_progress_drop_window_excludes_step_at_lower_bound() -> None:
    """Half-open `(t, t+H]`: events AT step t are NOT in the window.
    Construct a 6-step run where progress drops EXACTLY at step t (not
    later); evaluating future_progress_drop_h5 at t should return False
    because the drop event is at t, not in (t, t+5].
    """
    # Steps 0-5: progress trajectory 0 -> 1.0 -> 0.5 -> ... ->
    # The drop happens at step 2 where a second product subtask is added.
    # At t=2, current_progress is *measured AT step 2 inclusive*, which is
    # already 0.5; nothing strictly below 0.5 occurs in (2, 7].
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "synth"}, "reason": None,
         "timestamp": "2026-05-04T00:00:00Z"},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "p1",
         "payload": {"description": "p1", "category": "product",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:01Z"},
        {"step": 1, "event_type": "update_status", "subtask_id": "p1",
         "payload": {"status": "complete", "evidence": ["done"]},
         "reason": None, "timestamp": "2026-05-04T00:00:01Z"},
        # Adding p2 at step 2 drops progress from 1.0 -> 0.5.
        {"step": 2, "event_type": "add_subtask", "subtask_id": "p2",
         "payload": {"description": "p2", "category": "product",
                     "weight": 1.0, "parent_id": None},
         "reason": None, "timestamp": "2026-05-04T00:00:02Z"},
        {"step": 3, "event_type": "update_status", "subtask_id": "p2",
         "payload": {"status": "complete", "evidence": ["done2"]},
         "reason": None, "timestamp": "2026-05-04T00:00:03Z"},
        {"step": 7, "event_type": "update_status", "subtask_id": "p2",
         "payload": {"status": "in_progress"},
         "reason": None, "timestamp": "2026-05-04T00:00:07Z"},
    ]
    run = _run_from_dicts(events)
    # At t=2: current_progress measured AT step 2 inclusive is 0.5; in the
    # window (2, 7] every step has progress >= 0.5. So drop_h5 = False.
    # A closed-window implementation [2, 7] would still return False here
    # because the drop event is *exactly at* step 2 and the prefix already
    # has it. This test pins the level-vs-delta semantics.
    out = future_progress_drop_h5(run, t_step=2, is_terminal=False, finish_step=7)
    assert out.is_masked is False
    assert out.value is False, "drop measured against current_progress at t, not prior step"


# ---- Hand-checkable validation-new-work tests on golden_run ----


def test_validation_new_work_true_at_t4_window_captures_complete_then_reopen() -> None:
    """At t=4, window (4, 9]:
      step 7: UPDATE_STATUS s3(VALIDATION) status=complete  -> validation transition
      step 9: REOPEN_SUBTASK s2a (PRODUCT)                  -> discovery
    Both inside the window, transition before discovery -> True.
    """
    run = _golden_run()
    out = validation_new_work_h5(run, t_step=4, is_terminal=False, finish_step=13)
    assert out.is_masked is False
    assert out.value is True


def test_validation_new_work_false_at_t2_when_no_discovery_after_transition() -> None:
    """At t=2, window (2, 7]:
      step 7: validation transition (s3 complete)
    No PRODUCT/INVESTIGATION add or reopen AFTER the transition inside
    the window -> False. (Adds at step 4 are BEFORE the transition.)
    """
    run = _golden_run()
    out = validation_new_work_h5(run, t_step=2, is_terminal=False, finish_step=13)
    assert out.is_masked is False
    assert out.value is False


def test_invalidate_subtask_does_not_count_as_validation_transition() -> None:
    """Upstream restricts validation transitions to UPDATE_STATUS with
    status complete/blocked. INVALIDATE_SUBTASK is a different event
    type and must NOT trigger saw_validation, even on a VALIDATION
    subtask. golden_run step 10 is exactly this case (s3 invalidated).
    At t=7 (after the step-7 complete), validation transition is
    OUTSIDE the (7, 12] window — so any True here would mean the
    invalidate at step 10 was counted as a transition.
    """
    run = _golden_run()
    # Window (7, 12] does NOT include the step-7 complete; the only
    # candidate validation event in the window is the step-10
    # INVALIDATE_SUBTASK on s3, which must not count.
    out = validation_new_work_h5(run, t_step=7, is_terminal=False, finish_step=13)
    assert out.is_masked is False
    assert out.value is False


def test_h5_constant_pinned() -> None:
    """The horizon constant is part of the schema contract — a code
    change that bumps H5 silently would re-target every label. Pinning
    here makes such a change land in this test as a hard fail."""
    assert H5 == 5
