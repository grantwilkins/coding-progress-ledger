"""
Claim:
planner_latency.measure times the selection step alone, for the pure LP and the
pure greedy, against a target both can attain, and refuses to report a timing
that came from a fallback.

Plausible wrong implementations:
- Time the whole plan() call, so the shared table build, packing and deadline
  repair swamp the solver difference and can even invert it.
- Include the candidate-table build in the selection timing.
- Hide an exact MILP behind the greedy name.
- Ask for a target neither can reach, so the LP answers from its max-shed
  fallback and the comparison no longer measures the requested policies.
- Report the timing anyway when the selection missed the target.
- Report the first sample or the mean instead of the median over repeats.
"""

from __future__ import annotations

import time

import pytest

import planner_latency
import pool_planner


def test_selection_timing_excludes_the_shared_table_build(monkeypatch):
    build = pool_planner.candidate_table
    monkeypatch.setattr(pool_planner, "candidate_table",
                        lambda *a, **k: (time.sleep(.2), build(*a, **k))[1])

    row = planner_latency.measure(28, repeats=1)

    # The build is shared by both policies, so charging it to either one hides
    # the difference the measurement exists to show.
    assert row["table_build_s"] >= .2
    assert row["lp_select_s"] < .2 and row["greedy_select_s"] < .2


def test_selection_timing_is_the_selection(monkeypatch):
    solve = pool_planner._lp
    monkeypatch.setattr(pool_planner, "_lp",
                        lambda t, g: (time.sleep(.15), solve(t, g))[1])

    row = planner_latency.measure(28, repeats=1)

    assert row["lp_select_s"] >= .15
    assert row["greedy_select_s"] < .15


def test_greedy_never_reaches_for_the_exact_milp(monkeypatch):
    monkeypatch.setattr(pool_planner, "_integral_target_recovery",
                        lambda *a: pytest.fail("the greedy used the exact MILP"))

    planner_latency.measure(28, repeats=1)


def test_a_selection_that_misses_the_target_is_refused(monkeypatch):
    """A policy scored in its fallback is not that policy, so the run fails."""
    monkeypatch.setattr(pool_planner, "_lp", lambda table, target: set())

    with pytest.raises(RuntimeError, match="fallback"):
        planner_latency.measure(28, repeats=1)


def test_reported_time_is_the_median_not_the_first_or_the_mean(monkeypatch):
    solve, delays = pool_planner._lp, iter((.3, 0.0, 0.0))
    monkeypatch.setattr(pool_planner, "_lp",
                        lambda t, g: (time.sleep(next(delays)), solve(t, g))[1])

    row = planner_latency.measure(28, repeats=3)

    # First would report .3 and the mean .1; only the median reports ~0.
    assert row["lp_select_s"] < .05


def test_the_swept_target_is_attainable_by_both_policies():
    """If the default target were out of reach the LP would answer from its
    max-shed fallback, which is the flaw that made an earlier sweep report
    byte-identical plans for both policies."""
    row = planner_latency.measure(28, repeats=1)

    assert row["target_w"] > 0
    assert row["lp_moves"] and row["greedy_moves"]
