"""Prefix replay engine: the load-bearing future-leakage test.

Claim:
    prefix_replay(run, t) returns a ReplayState whose events_so_far has
    every event with step <= t and no event with step > t. The replayed
    ledger and progress observation are functions of events_so_far only;
    mutating events past t in the run cannot change either at any
    t' <= t.

Plausible wrong implementations:
    - off-by-one filter (`e.step < t` instead of `e.step <= t`) drops the
      events at the checkpoint step itself
    - filter applied to a copy but events_so_far returned from the
      original list -> future events leak
    - ledger replayed against the FULL run, then filtered after -> state
      reflects future events at every t
    - score() called on the run-level ledger, not the prefix ledger
    - missing FutureLeakageError so silent corruption survives
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ledger_progress.core import EventType, LedgerEvent
from ledger_progress.serialization import load_events_jsonl

from coding_estimator.checkpoints.replay import (
    FutureLeakageError,
    ReplayState,
    _assert_prefix_only,
    prefix_replay,
)
from coding_estimator.ingest.run_record import RunRecord

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden_run"
CANONICAL = FIXTURE_DIR / "ledger.jsonl"
MUTATED = FIXTURE_DIR / "ledger_mutated_after_t_mid.jsonl"
EXPECTED = json.loads((FIXTURE_DIR / "expected_checkpoints.json").read_text())


def _run(jsonl: Path) -> RunRecord:
    """Build a RunRecord directly from a fixture jsonl. We do NOT go
    through load_run because that path requires a runs_dir under the
    upstream ledger checkout; the fixture lives inside our repo."""
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


def test_prefix_replay_at_every_step_matches_expected_aggregates() -> None:
    """For every t from 0 to terminal, the prefix replay's coding_score
    must match the hand-derived expected_checkpoints.json values."""
    run = _run(CANONICAL)
    expected_by_step = {c["step"]: c for c in EXPECTED["checkpoints"]}
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        exp = expected_by_step[t]
        assert state.coding_score.active_leaf_count == exp["active_leaf_count"], t
        assert state.coding_score.complete_leaf_count == exp["completed_leaf_count"], t
        assert abs(state.coding_score.progress - exp["coding_progress"]) < 1e-9, t


def test_events_so_far_contains_only_prefix() -> None:
    run = _run(CANONICAL)
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        assert all(e.step <= t for e in state.events_so_far), t


def test_events_so_far_includes_events_at_checkpoint_step() -> None:
    """Off-by-one regression: a buggy filter using `<` instead of `<=`
    would drop the events emitted exactly at step t."""
    run = _run(CANONICAL)
    # Step 1 in the canonical fixture has TWO events: add_subtask s1
    # and update_status s1 in_progress. Both must be in the prefix at t=1.
    state = prefix_replay(run, 1)
    step_1_events = [e for e in state.events_so_far if e.step == 1]
    assert len(step_1_events) == 2, "off-by-one: events at step==t got dropped"


def test_future_mutation_invariance_at_every_t_le_t_mid() -> None:
    """For every t <= t_mid, prefix_replay on the canonical and mutated
    fixtures must produce IDENTICAL events_so_far and IDENTICAL scoring
    observations. Past t_mid the two fixtures diverge by design.
    This is the load-bearing leakage property."""
    canonical = _run(CANONICAL)
    mutated = _run(MUTATED)
    t_mid = EXPECTED["t_mid"]
    for t in range(0, t_mid + 1):
        a = prefix_replay(canonical, t)
        b = prefix_replay(mutated, t)
        # event-by-event equality on the prefix
        assert len(a.events_so_far) == len(b.events_so_far), t
        for ea, eb in zip(a.events_so_far, b.events_so_far, strict=True):
            assert ea.step == eb.step
            assert ea.event_type == eb.event_type
            assert ea.subtask_id == eb.subtask_id
            assert ea.payload == eb.payload
        # score equality
        assert a.coding_score.progress == b.coding_score.progress, t
        assert a.coding_score.active_leaf_count == b.coding_score.active_leaf_count, t
        assert a.coding_score.complete_leaf_count == b.coding_score.complete_leaf_count, t


def test_canonical_and_mutated_diverge_past_t_mid() -> None:
    """Sanity: at terminal step the two fixtures MUST differ. Otherwise
    the future-mutation test above is asserting against a no-op."""
    canonical = _run(CANONICAL)
    mutated = _run(MUTATED)
    t_terminal = EXPECTED["terminal_step"]
    a = prefix_replay(canonical, t_terminal)
    b = prefix_replay(mutated, t_terminal)
    assert a.coding_score.progress != b.coding_score.progress or (
        a.coding_score.active_leaf_count != b.coding_score.active_leaf_count
    )


def test_future_leakage_error_fires() -> None:
    """A hand-poisoned event list with step past t_step must trigger
    the FutureLeakageError. This guards against a future regression
    where the filter is dropped but the assertion still fires."""
    bogus = (
        LedgerEvent(step=0, event_type=EventType.INIT, subtask_id=None, payload={}),
        LedgerEvent(step=5, event_type=EventType.INIT, subtask_id=None, payload={}),
    )
    with pytest.raises(FutureLeakageError, match="step=5"):
        _assert_prefix_only(bogus, t_step=2)


def test_negative_t_step_rejected() -> None:
    run = _run(CANONICAL)
    with pytest.raises(ValueError, match="non-negative"):
        prefix_replay(run, -1)


def test_replay_state_is_purely_a_function_of_inputs() -> None:
    """Calling prefix_replay twice with the same inputs must produce
    equal observable state. Catches any global mutable state lurking in
    the engine."""
    run = _run(CANONICAL)
    a = prefix_replay(run, 7)
    b = prefix_replay(run, 7)
    assert isinstance(a, ReplayState) and isinstance(b, ReplayState)
    assert a.events_so_far == b.events_so_far
    assert a.coding_score.progress == b.coding_score.progress
    assert sorted(a.ledger.subtasks) == sorted(b.ledger.subtasks)


def test_terminal_step_replay_uses_all_events() -> None:
    run = _run(CANONICAL)
    state = prefix_replay(run, EXPECTED["terminal_step"])
    assert len(state.events_so_far) == len(run.events)
