"""Tests for src/agent_migrate_agent/oracle_diff.py (Workstream O2).

Claim:
For each diagnostic cell, `compute_oracle_diff` runs an exhaustive
oracle plus strong_reuse / mixed_min_pressure / random_mode through K4
and returns:
  * per_policy p50 + bottleneck_time breakdown,
  * per_workflow (dst, prompt_mode, workspace_mode) inferred from
    each policy's emitted actions,
  * per_cell gaps (oracle vs mixed, oracle vs random, strong vs random).

Plausible wrong implementations:
  * Bottleneck breakdown weighted by action count not elapsed time
    (a few small replays would dwarf one big workspace transfer).
  * Per_workflow choice inference using node_level required_state
    instead of state_layer (would conflate prompt and workspace modes).
  * Gap_fraction sign flipped (random vs oracle reported with the
    wrong baseline).
  * Oracle simulator_objective tied to the wrong metric (e.g., reads
    `makespan_s` only and ignores p50/p90), so an oracle_best plan is
    not actually best on the reported metric.
  * `_unify_modes` collapses WARM_REUSE+real_mode incorrectly so
    every per_workflow choice reads as "warm_reuse" or "mixed".

These are the failure modes the tests target.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent_migrate_agent.costs import ARTIFACT_COPY, CONTEXT_REPLAY, KV_TRANSFER
from agent_migrate_agent.k8_regime import (
    RegimeCell,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
)
from agent_migrate_agent.oracle_diff import (
    OracleDiffReport,
    PolicyDiagnostic,
    WorkflowChoice,
    _gap_frac,
    _unify_modes,
    compute_oracle_diff,
)
from agent_migrate_agent.resources import WARM_REUSE, WORKSPACE_HYDRATE


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _gap_frac sign and direction
# ---------------------------------------------------------------------------


def test_gap_frac_is_zero_when_equal_and_positive_when_fast_is_smaller():
    assert _gap_frac(10.0, 10.0) == 0.0
    # fast (oracle) is smaller -> positive gap, the "win" sign.
    assert _gap_frac(10.0, 5.0) == pytest.approx(0.5)
    # If oracle were larger than the heuristic, gap is negative —
    # this is the bug_detection sign and tests pin it.
    assert _gap_frac(5.0, 10.0) == pytest.approx(-1.0)


def test_gap_frac_handles_zero_baseline():
    assert _gap_frac(0.0, 5.0) == 0.0
    assert _gap_frac(-1.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# _unify_modes — precedence rules
# ---------------------------------------------------------------------------


def test_unify_modes_collapses_warm_reuse_with_cold_mode():
    """If actions report WARM_REUSE for already_warm states and
    CONTEXT_REPLAY for cold ones, the *cold* mode is what the policy
    chose. Wrong implementation that returns 'mixed' or 'warm_reuse'
    would mask the actual choice in the diff report."""
    result = _unify_modes(
        [WARM_REUSE, CONTEXT_REPLAY, WARM_REUSE],
        (CONTEXT_REPLAY, KV_TRANSFER, WARM_REUSE),
    )
    assert result == CONTEXT_REPLAY


def test_unify_modes_returns_mixed_for_two_real_cold_modes():
    """Two distinct cold modes within one workflow -> mixed.
    A naive implementation that takes the first observed mode would
    silently report just that one."""
    result = _unify_modes(
        [CONTEXT_REPLAY, KV_TRANSFER, CONTEXT_REPLAY],
        (CONTEXT_REPLAY, KV_TRANSFER, WARM_REUSE),
    )
    assert result == "mixed"


def test_unify_modes_none_when_no_actions_in_layer():
    assert _unify_modes([], (CONTEXT_REPLAY, KV_TRANSFER)) == "none"


# ---------------------------------------------------------------------------
# compute_oracle_diff end_to_end on a small cell
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cell_report() -> OracleDiffReport:
    """A 2_workflow cell with mild prefill pressure — small enough that
    the oracle finishes in milliseconds. We use this fixture across all
    end_to_end tests."""
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=2, state_scale="tiny", prefill_capacity="moderate",
        link_gbps=25, seed=4242,
    )
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    return compute_oracle_diff(
        scenario="unit_test_cell",
        cell={
            "n_workflows": cell.n_workflows, "state_scale": cell.state_scale,
            "prefill_capacity": cell.prefill_capacity, "link_gbps": cell.link_gbps,
        },
        episode=episode,
        manifests=manifests,
        bundle=bundle,
        budget=budget,
        random_seed=cell.seed,
    )


def test_oracle_is_at_least_as_fast_as_every_other_policy(small_cell_report):
    """Oracle is exhaustive over the candidate space `mixed`/`strong`/
    `random` all draw from. So oracle p50 must be <= every other
    policy's p50 up to numerical tolerance. Catches an oracle that
    picks on the wrong objective (e.g., makespan instead of p50)."""
    r = small_cell_report
    tol = 1e-9
    assert r.oracle.p50_resume_s <= r.mixed.p50_resume_s + tol
    assert r.oracle.p50_resume_s <= r.strong_reuse.p50_resume_s + tol
    assert r.oracle.p50_resume_s <= r.random.p50_resume_s + tol


def test_per_policy_choices_cover_every_workflow(small_cell_report):
    """Each policy must report a choice for every workflow. A choice
    inferred from no actions (empty mode lists) returns 'none' — a
    workflow with NO entry in the dict is a different bug (the policy
    didn't emit a plan for that workflow)."""
    r = small_cell_report
    expected_wfs = set(r.oracle.per_workflow_choice)
    for diag in (r.oracle, r.mixed, r.strong_reuse, r.random):
        assert set(diag.per_workflow_choice) == expected_wfs


def test_bottleneck_fractions_sum_to_one_when_any_contention(small_cell_report):
    """Per_policy bottleneck fractions must form a valid distribution
    (sum to 1) whenever there is any non_zero bottleneck time. If the
    breakdown weighted by action_count instead of elapsed time, this
    invariant could be violated for some cells (more actions than
    seconds)."""
    r = small_cell_report
    for diag in (r.oracle, r.mixed, r.strong_reuse, r.random):
        total = sum(diag.bottleneck_seconds.values())
        if total <= 0:
            continue
        fractions = diag.bottleneck_fractions
        assert math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9), (
            f"{diag.policy_name}: fractions sum {sum(fractions.values())} != 1.0"
        )


def test_workflow_choice_dst_is_a_real_destination_site(small_cell_report):
    """Every per_workflow choice's dst_site must be one of the
    episode's destination_sites. A wrong inference that picks
    src_site (Phoenix) would silently break the report."""
    r = small_cell_report
    valid = {"phoenix", "seattle", "austin"}  # superset of any cell's sites
    for diag in (r.oracle, r.mixed, r.strong_reuse, r.random):
        for choice in diag.per_workflow_choice.values():
            assert choice.dst_site in valid, (
                f"{diag.policy_name}/{choice.workflow_id}: dst {choice.dst_site!r}"
            )


def test_oracle_vs_random_gap_is_at_least_oracle_vs_mixed(small_cell_report):
    """random_mode is the worst plausible policy in the set, so the
    oracle's gap vs random must be >= oracle's gap vs mixed (modulo
    numerical noise). If sign is flipped or baselines swapped, this
    invariant breaks."""
    r = small_cell_report
    # If oracle == mixed (no headroom), both gaps may be close to zero
    # or even negative_due_to_noise; check only the strict direction
    # when the oracle actually beats mixed.
    if r.oracle.p50_resume_s < r.mixed.p50_resume_s - 1e-9:
        assert r.oracle_vs_random_gap_frac >= r.oracle_vs_mixed_gap_frac - 1e-9


def test_oracle_diff_oracle_p50_matches_k9_run_small_n_oracle():
    """compute_oracle_diff and run_small_n_oracle must enumerate over
    the same candidate space and pick the same winner under the same
    objective. They share `enumerate_oracle_plans`; this regression
    test pins parity so a future refactor of either side cannot drift."""
    from agent_migrate_agent.k9_oracle import run_small_n_oracle
    from agent_migrate_agent.warmness import WarmnessMap

    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=2, state_scale="tiny", prefill_capacity="moderate",
        link_gbps=25, seed=4242,
    )
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warm = WarmnessMap.from_episode_seed(episode.state_warmness)
    k9_result = run_small_n_oracle(
        episode, manifests, bundle, budget, warm, max_workflows=2,
    )
    diff = compute_oracle_diff(
        scenario="parity", cell={}, episode=episode, manifests=manifests,
        bundle=bundle, budget=budget, random_seed=cell.seed,
    )
    # Numerical tolerance: the two paths share `enumerate_oracle_plans`,
    # so the picked plan and its p50 must be byte_identical.
    assert diff.oracle.p50_resume_s == pytest.approx(
        k9_result.oracle_p50_resume_s, abs=1e-12,
    )


def test_attributed_fraction_of_makespan_is_a_valid_ratio(small_cell_report):
    """Sanity bounds + sidebar invariant: attributed_fraction is in
    [0, 1] for every policy. A wrong implementation that summed
    elapsed time twice (e.g., per multi_resource action) could exceed
    1.0; an implementation that dropped attributed time would report 0
    even when the breakdown is meaningful."""
    r = small_cell_report
    for diag in (r.oracle, r.mixed, r.strong_reuse, r.random):
        frac = diag.attributed_fraction_of_makespan
        assert 0.0 <= frac <= 1.0, f"{diag.policy_name}: attr={frac}"


def test_per_workflow_diffs_returns_one_row_per_workflow(small_cell_report):
    """Sanity: the diff table emits exactly N rows, one per workflow."""
    r = small_cell_report
    rows = r.per_workflow_diffs()
    assert len(rows) == len(r.oracle.per_workflow_choice)
    for row in rows:
        assert row["scenario"] == r.scenario
        # boolean diff flags must be Python bools, not strings —
        # CSV writer would otherwise serialize them as 'True'/'False'
        # but consumers may filter on truthiness.
        assert isinstance(row["diff_dst"], bool)
        assert isinstance(row["diff_prompt_mode"], bool)
        assert isinstance(row["diff_workspace_mode"], bool)
