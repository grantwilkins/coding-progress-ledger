"""D3a-d feature builders pinned against the D0 golden fixture.

Claim:
    For every step t in [0, terminal] of the canonical golden fixture,
    the four simple-aggregate feature builders (frontier, closure,
    discovery, instability) produce values that match
    expected_checkpoints.json at every column it covers. AND for every
    t <= t_mid, the same builders produce IDENTICAL outputs on the
    canonical and mutated fixtures (future-mutation invariance per
    AGENTS.md invariant 2).

Plausible wrong implementations:
    - frontier counts incomplete leaves only (omits complete leaves)
    - closure progress denominator excludes invalidated leaves but
      numerator does not (introduces > 1.0 progress)
    - discovery `num_adds_so_far` counts SPLIT children too -> over-
      count; or `denominator_growth_so_far` counts only ADD events ->
      under-count
    - instability `num_progress_drops_so_far` ignores the leading edge
      (the 0->non-zero transition at step 1 should not count as a drop)
    - instability `largest_progress_drop_so_far` returns the most
      RECENT drop instead of the largest seen so far
    - any builder reads run.events directly instead of state.events_so_far
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ledger_progress.serialization import load_events_jsonl

from coding_estimator.checkpoints.features import (
    closure,
    discovery,
    frontier,
    instability,
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
        arm=None,
        difficulty=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


def test_frontier_active_leaf_count_matches_expected_at_every_step() -> None:
    run = _run(CANONICAL)
    by_step = {c["step"]: c for c in EXPECTED["checkpoints"]}
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        out = frontier.compute(state)
        assert out["active_leaf_count"] == by_step[t]["active_leaf_count"], t


def test_closure_completed_count_and_coding_progress_match_expected() -> None:
    run = _run(CANONICAL)
    by_step = {c["step"]: c for c in EXPECTED["checkpoints"]}
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        out = closure.compute(state)
        assert out["completed_leaf_count"] == by_step[t]["completed_leaf_count"], t
        assert abs(out["coding_progress"] - by_step[t]["coding_progress"]) < 1e-9, t


def test_closure_progress_in_unit_interval() -> None:
    """Per-category progress must be in [0, 1] at every checkpoint;
    a mismatched numerator/denominator would surface as a value > 1."""
    run = _run(CANONICAL)
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        out = closure.compute(state)
        for col in (
            "coding_progress",
            "validation_progress",
            "product_progress",
            "investigation_progress",
        ):
            assert 0.0 <= out[col] <= 1.0, (t, col, out[col])


def test_discovery_counts_match_expected() -> None:
    run = _run(CANONICAL)
    by_step = {c["step"]: c for c in EXPECTED["checkpoints"]}
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        out = discovery.compute(state)
        assert out["num_adds_so_far"] == by_step[t]["num_adds_so_far"], t
        assert out["num_splits_so_far"] == by_step[t]["num_splits_so_far"], t


def test_discovery_denominator_growth_distinguishes_adds_from_splits() -> None:
    """At step 5, the split adds 2 children. denominator_growth must
    reflect adds (3) + split-children (2) = 5; num_adds_so_far stays 3.
    A buggy implementation that counts adds only, or that counts the
    split as one event, would diverge here."""
    run = _run(CANONICAL)
    state = prefix_replay(run, 5)
    out = discovery.compute(state)
    assert out["num_adds_so_far"] == 3
    assert out["num_splits_so_far"] == 1
    assert out["denominator_growth_so_far"] == 5


def test_discovery_window_features_strictly_local() -> None:
    """new_leaf_count_last_K_steps must consider events with step in
    (t-K, t]. At t=2, only the add at step 1 is in the last-1-step
    window; the larger windows should also see the step-1 add."""
    run = _run(CANONICAL)
    state = prefix_replay(run, 2)
    out = discovery.compute(state)
    # last_1_steps = (1, 2] -> events at step 2 only -> 0 adds.
    # last_3_steps = (-1, 2] -> events at steps 0,1,2 -> 1 add (step 1).
    # last_5_steps = (-3, 2] -> 1 add.
    assert out["new_leaf_count_last_1_steps"] == 0
    assert out["new_leaf_count_last_3_steps"] == 1
    assert out["new_leaf_count_last_5_steps"] == 1


def test_instability_counts_and_drops_match_expected() -> None:
    run = _run(CANONICAL)
    by_step = {c["step"]: c for c in EXPECTED["checkpoints"]}
    for t in range(0, EXPECTED["terminal_step"] + 1):
        state = prefix_replay(run, t)
        out = instability.compute(state, run)
        assert out["num_reopens_so_far"] == by_step[t]["num_reopens_so_far"], t
        assert (
            out["num_invalidations_so_far"]
            == by_step[t]["num_invalidations_so_far"]
        ), t
        assert (
            out["num_progress_drops_so_far"]
            == by_step[t]["num_progress_drops_so_far"]
        ), t
        assert (
            abs(
                out["largest_progress_drop_so_far"]
                - by_step[t]["largest_progress_drop_so_far"]
            )
            < 1e-9
        ), t


def test_instability_largest_drop_is_largest_not_latest() -> None:
    """At t=10, the largest drop seen so far is 0.5 (at step 3 when s2
    was added). Later drops at steps 9 and 10 are smaller (0.25 and
    ~0.167). A buggy implementation that returns the LATEST drop
    instead of the LARGEST would return ~0.167 at t=10."""
    run = _run(CANONICAL)
    state = prefix_replay(run, 10)
    out = instability.compute(state, run)
    assert out["largest_progress_drop_so_far"] == 0.5


@pytest.mark.parametrize(
    "builder_name,builder",
    [
        ("frontier", lambda s, r: frontier.compute(s)),
        ("closure", lambda s, r: closure.compute(s)),
        ("discovery", lambda s, r: discovery.compute(s)),
        ("instability", lambda s, r: instability.compute(s, r)),
    ],
)
def test_future_mutation_invariance_for_each_builder(builder_name, builder) -> None:
    """For every t <= t_mid, builder output on canonical and mutated
    fixtures must be identical. This is the load-bearing invariant
    that AGENTS.md item 2 codifies for every feature group."""
    canonical = _run(CANONICAL)
    mutated = _run(MUTATED)
    t_mid = EXPECTED["t_mid"]
    for t in range(0, t_mid + 1):
        a = builder(prefix_replay(canonical, t), canonical)
        b = builder(prefix_replay(mutated, t), mutated)
        assert a == b, f"{builder_name} diverged at t={t}: {a} != {b}"
