"""D3e-h: stalling, validation, evidence, time_budget builders.

Each builder is exercised on the D0 golden fixture for:
    1. Hand-checkable spot values at chosen steps.
    2. Future-mutation invariance for every t <= t_mid.

We do NOT pin every column at every step (the fixture only carries
hand-derived values for the simple-aggregate groups). Spot tests
target the dangerous semantic mistakes per builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ledger_progress.serialization import load_events_jsonl

from coding_estimator.checkpoints.features import (
    evidence,
    stalling,
    time_budget,
    validation,
)
from coding_estimator.checkpoints.replay import prefix_replay
from coding_estimator.ingest.run_record import RunRecord

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_run"
CANONICAL = FIXTURE_DIR / "ledger.jsonl"
MUTATED = FIXTURE_DIR / "ledger_mutated_after_t_mid.jsonl"
EXPECTED = json.loads((FIXTURE_DIR / "expected_checkpoints.json").read_text())


def _run(jsonl: Path) -> RunRecord:
    events = tuple(load_events_jsonl(str(jsonl)))
    return RunRecord(
        run_id="golden",
        source="tb_live",
        ledger_path=jsonl,
        events=events,
        has_real_wallclock=True,
        start_wall_time=None,
        end_wall_time=None,
        task_id="golden",
        task_family=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


# --- stalling -----------------------------------------------------------

def test_stalling_blocked_count_zero_until_step_11() -> None:
    """In the canonical fixture nothing is BLOCKED until step 11
    (s2b -> blocked). Before that, blocked_*_count must be 0; at
    step 11+ the validation/coding count fires."""
    run = _run(CANONICAL)
    for t in (5, 9, 10):
        out = stalling.compute(prefix_replay(run, t))
        assert out["blocked_leaf_count"] == 0, t
    out11 = stalling.compute(prefix_replay(run, 11))
    assert out11["blocked_leaf_count"] == 1
    # s2b is product; no validation blocked (s3 was invalidated, not blocked).
    assert out11["blocked_validation_leaf_count"] == 0
    assert out11["blocked_coding_leaf_count"] == 1


def test_stalling_steps_since_completion_tracks_latest_complete() -> None:
    """At step 7 (after s3 completes), steps_since_completion must be 0.
    At step 8 (add_evidence only, no completion), it must be 1."""
    run = _run(CANONICAL)
    out7 = stalling.compute(prefix_replay(run, 7))
    out8 = stalling.compute(prefix_replay(run, 8))
    assert out7["steps_since_completion"] == 0
    assert out8["steps_since_completion"] == 1


def test_stalling_no_progress_window_false_when_progress_changes() -> None:
    """At step 7, coding_progress just rose to 0.75. The trailing 5-step
    window includes the step 6 increase (0.25 -> 0.5) so
    no_progress_window_5 must be False."""
    run = _run(CANONICAL)
    out = stalling.compute(prefix_replay(run, 7))
    assert out["no_progress_window_5"] is False


# --- validation ---------------------------------------------------------

def test_validation_no_signal_before_first_validation_subtask() -> None:
    """Before step 4 (when s3, the validation subtask, is added), no
    validation features should fire. submit_without_validation_so_far
    must be True at every t < 4."""
    run = _run(CANONICAL)
    for t in (0, 1, 2, 3):
        out = validation.compute(prefix_replay(run, t))
        assert out["validation_leaf_exists"] is False, t
        assert out["validation_started"] is False, t
        assert out["validation_complete"] is False, t
        assert out["submit_without_validation_so_far"] is True, t


def test_validation_pass_then_fail_state_transition() -> None:
    """At step 7 (s3 complete) the validation pass fires; at step 10
    (s3 invalidated) validation_failed becomes True and
    validation_complete becomes False (s3 is no longer COMPLETE)."""
    run = _run(CANONICAL)
    out7 = validation.compute(prefix_replay(run, 7))
    out10 = validation.compute(prefix_replay(run, 10))
    assert out7["validation_complete"] is True
    assert out7["validation_failed"] is False
    assert out10["validation_complete"] is False
    assert out10["validation_failed"] is True


def test_submit_without_validation_so_far_flips_at_step_4() -> None:
    """At step 3 there's no validation subtask yet; at step 4 the
    validation add fires, so the flag must flip from True to False."""
    run = _run(CANONICAL)
    pre = validation.compute(prefix_replay(run, 3))
    post = validation.compute(prefix_replay(run, 4))
    assert pre["submit_without_validation_so_far"] is True
    assert post["submit_without_validation_so_far"] is False


def test_validation_attempts_and_successes_count_separately() -> None:
    """num_validation_attempts counts every UPDATE_STATUS on a validation
    leaf; num_validation_successes counts only those whose payload is
    `complete`. At step 7 there should be 2 attempts (in_progress,
    complete) and 1 success. A wrong implementation that conflates them
    would report attempts==successes."""
    run = _run(CANONICAL)
    out = validation.compute(prefix_replay(run, 7))
    assert out["num_validation_attempts"] == 2
    assert out["num_validation_successes"] == 1


# --- evidence -----------------------------------------------------------

def test_evidence_completion_counts_match_fixture() -> None:
    """At step 2 (s1 complete with evidence ['repro confirmed']),
    classify_evidence sees 'contract' in 'repro confirmed'? No -- but
    no evidence pattern matches, so the result is {manual_note}
    -> manual_only completion. At step 6 (s2a 'A merged') the word
    'merged' isn't a strong term either; that's also manual_only.
    But at step 7 (s3 'unit test green') the word 'test' matches
    test_output OR similar terms? Actually 'unit test green' contains
    no patterns directly. Strong fraction may be low; what we pin is
    that strong + manual_only equals total completions."""
    run = _run(CANONICAL)
    out = evidence.compute(prefix_replay(run, 13))
    # Five completions in the canonical fixture (steps 2, 6, 7, 12, 13).
    total = (
        out["strong_completion_count"]
        + out["manual_only_completion_count"]
    )
    # Every completion must classify as either strong or manual_only.
    assert total == 5


def test_evidence_classifies_test_output_as_strong() -> None:
    """A completion event whose evidence string contains 'pytest passed'
    must classify as strong. This is a direct check on the upstream
    snapshot's contract."""
    from coding_estimator.checkpoints.features._upstream_evidence_snapshot import (
        STRONG_EVIDENCE_TYPES,
        classify_evidence,
    )

    types = classify_evidence(["pytest tests passed in 0.3s"])
    assert types & STRONG_EVIDENCE_TYPES


def test_evidence_classifies_pure_natural_language_as_manual_only() -> None:
    from coding_estimator.checkpoints.features._upstream_evidence_snapshot import (
        classify_evidence,
    )

    types = classify_evidence(["I think this works"])
    assert types == {"manual_note"}


# --- time_budget --------------------------------------------------------

def test_time_budget_elapsed_steps_is_t_minus_smin() -> None:
    run = _run(CANONICAL)
    for t in (0, 5, 13):
        out = time_budget.compute(prefix_replay(run, t), run)
        assert out["elapsed_steps"] == t  # s_min == 0 in this fixture


def test_time_budget_wallclock_populated_when_real_wallclock() -> None:
    """The fixture's events have a 1-second cadence between steps. At
    t=5, elapsed_wall_time must be > 0 (and roughly 5 seconds, but we
    don't pin exact)."""
    run = _run(CANONICAL)
    out = time_budget.compute(prefix_replay(run, 5), run)
    assert out["elapsed_wall_time"] is not None
    assert out["elapsed_wall_time"] > 0


def test_time_budget_wallclock_null_when_no_real_wallclock() -> None:
    """A run without real wallclock must NOT silently fabricate
    elapsed_wall_time = 0; it must be None."""
    run_no_wc = RunRecord(
        run_id="r",
        source="swe_agent_pilot",
        ledger_path=CANONICAL,
        events=tuple(load_events_jsonl(str(CANONICAL))),
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id=None,
        task_family=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )
    out = time_budget.compute(prefix_replay(run_no_wc, 5), run_no_wc)
    assert out["elapsed_wall_time"] is None


def test_time_budget_timeout_features_null_until_d4_wires_in_timeout() -> None:
    """fraction_timeout_consumed and remaining_timeout_budget require
    a per-task timeout that v0 doesn't have a path to. Until D4 wires
    that in, both features must be None on every run, NEVER 0.0."""
    run = _run(CANONICAL)
    out = time_budget.compute(prefix_replay(run, 5), run)
    assert out["fraction_timeout_consumed"] is None
    assert out["remaining_timeout_budget"] is None


# --- future-mutation invariance ----------------------------------------

@pytest.mark.parametrize(
    "builder_name,builder",
    [
        ("stalling", lambda s, r: stalling.compute(s)),
        ("validation", lambda s, r: validation.compute(s)),
        ("evidence", lambda s, r: evidence.compute(s)),
        ("time_budget", lambda s, r: time_budget.compute(s, r)),
    ],
)
def test_future_mutation_invariance_for_each_complex_builder(builder_name, builder) -> None:
    canonical = _run(CANONICAL)
    mutated = _run(MUTATED)
    t_mid = EXPECTED["t_mid"]
    for t in range(0, t_mid + 1):
        a = builder(prefix_replay(canonical, t), canonical)
        b = builder(prefix_replay(mutated, t), mutated)
        assert a == b, f"{builder_name} diverged at t={t}: {a} != {b}"
