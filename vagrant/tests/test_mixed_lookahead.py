from __future__ import annotations

"""
Claim:
`mixed_lookahead` is a one-step-lookahead reconstitution policy that scores
each (dst, prompt_mode) candidate at workflow w_i by
`max(immediate_max_pressure, post_next_workflow_max_pressure)`. It is a real
heuristic (not a no-op refactor), bounded above by the oracle, and reduces
to `mixed_min_pressure` when there is only one workflow.

Empirical claim — explicitly NEGATIVE for P50:
On the four O2 diagnostic cells, `mixed_lookahead` does NOT close the 35-50%
P50 gap. That is recorded as a load-bearing finding: the heuristic optimizes
a max-pressure proxy that corresponds to MAKESPAN, while the oracle wins on
P50 by deliberately unbalancing the herd. We pin a *makespan* improvement
on at least one cell, and document the P50 non-improvement explicitly so a
future regression that flips the ordering is detected.

Plausible wrong implementations:
- Lookahead is a no-op refactor: identical plans to mixed_min_pressure on
  every cell (lookahead score collapses to immediate score in degenerate
  ways).
- Lookahead violates the oracle bound: lookahead p50 < oracle p50 (= bug
  in candidate space, simulator, or budget).
- Lookahead breaks single-workflow case: with N=1 there is no "next
  workflow"; the policy must produce identical actions to mixed_min_pressure.
- Lookahead breaks the all-warm case: every action should be WARM_REUSE,
  matching strong reuse exactly.
- Lookahead picks dsts not in `episode.destination_sites` (programmer error
  on dst-tracking).
- Future "improvement" silently flips makespan ordering: a regression that
  makes lookahead worse than mixed on every cell would mask the negative
  finding; we pin one cell where lookahead makespan ≤ mixed makespan.
"""

from pathlib import Path

import pytest

from vagrant_agent.fluid_sim import simulate_fluid
from vagrant_agent.k8_regime import (
    RegimeCell,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
)
from vagrant_agent.k9_oracle import run_small_n_oracle
from vagrant_agent.reconstitution import (
    cache_reuse,
    mixed_lookahead,
    mixed_min_pressure,
)
from vagrant_agent.warmness import WarmnessMap


REPO = Path(__file__).resolve().parent.parent

# Same diagnostic cells as scripts/run_o2.py.
O2_CELLS = (
    ("tiny_prefill_pressure",       "tiny",     "tight", 100),
    ("medium_multi_resource",       "medium",   "tight",   5),
    ("monorepo_workspace_pressure", "monorepo", "loose", 100),
    ("slow_link_network_pressure",  "medium",   "loose",   1),
)


def _build_o2_inputs(scenario_index: int):
    name, state_scale, prefill_capacity, link_gbps = O2_CELLS[scenario_index]
    cell = RegimeCell(
        n_workflows=4,
        state_scale=state_scale,
        prefill_capacity=prefill_capacity,
        link_gbps=link_gbps,
        seed=9009 + scenario_index,
    )
    bundle = default_bundle(REPO)
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    return name, episode, manifests, bundle, budget, warmness


def _result(plan, episode, manifests, bundle, warmness, budget):
    return simulate_fluid(episode, manifests, plan, bundle, warmness, budget)


# ---------------------------------------------------------------------------
# Mechanical correctness — these are what makes the lookahead a real policy.
# ---------------------------------------------------------------------------


def test_lookahead_reduces_to_mixed_when_only_one_workflow():
    """Claim (boundary): with N=1 the lookahead has no successor to peek
    at; mixed_lookahead and mixed_min_pressure must produce byte-identical
    plans. A buggy lookahead that adds spurious score noise would diverge
    here even when there is no decision to make."""
    cell = RegimeCell(
        n_workflows=1,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=5,
        seed=7011,
    )
    bundle = default_bundle(REPO)
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    mixed = mixed_min_pressure(episode, manifests, bundle, warmness, budget)
    look = mixed_lookahead(episode, manifests, bundle, warmness, budget)
    assert set(mixed) == set(look)
    for wf_id in mixed:
        m_choices = [(a.state_id, a.mode, a.dst_site) for a in mixed[wf_id]]
        l_choices = [(a.state_id, a.mode, a.dst_site) for a in look[wf_id]]
        assert m_choices == l_choices


def test_lookahead_picks_actions_in_episode_destination_sites():
    """Claim (admissibility invariant): every action's dst_site is one of
    `episode.destination_sites`. A bug that lets the lookahead return a
    destination from `source_sites` (or a typo) silently violates the
    episode's contract."""
    cell = RegimeCell(
        n_workflows=4,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=5,
        seed=7022,
    )
    bundle = default_bundle(REPO)
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    warmness = WarmnessMap.from_episode_seed(episode.state_warmness)
    plan = mixed_lookahead(episode, manifests, bundle, warmness, budget)
    valid_dsts = set(episode.destination_sites)
    for actions in plan.values():
        for action in actions:
            assert action.dst_site in valid_dsts


def test_lookahead_matches_strong_reuse_when_all_warm():
    """Claim (warmness invariant): if every state is warm at every
    destination, every action is WARM_REUSE and the lookahead's plan
    is observationally identical to strong reuse — same p50 to numerical
    noise."""
    cell = RegimeCell(
        n_workflows=3,
        state_scale="tiny",
        prefill_capacity="tight",
        link_gbps=5,
        seed=7033,
    )
    bundle = default_bundle(REPO)
    episode, manifests = make_k8_episode(cell)
    budget = make_k8_budget(cell)
    all_sites = tuple(sorted(set(episode.source_sites) | set(episode.destination_sites)))
    warmness_seed = {}
    for manifest in manifests.values():
        for sid in manifest.state_objects:
            warmness_seed[sid] = all_sites
    warmness = WarmnessMap.from_episode_seed(warmness_seed)
    p50_strong = _result(
        cache_reuse(episode, manifests, bundle, warmness, budget),
        episode, manifests, bundle, warmness, budget,
    ).p50_resume_s()
    p50_look = _result(
        mixed_lookahead(episode, manifests, bundle, warmness, budget),
        episode, manifests, bundle, warmness, budget,
    ).p50_resume_s()
    assert p50_strong == pytest.approx(p50_look, abs=1e-9)


# ---------------------------------------------------------------------------
# Bounded above by oracle — admissibility of the candidate space.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_index", range(len(O2_CELLS)))
def test_lookahead_bounded_above_by_oracle_on_every_o2_cell(scenario_index):
    """Claim (admissibility): the K9 oracle exhaustively enumerates the
    same `(dst, prompt_mode, workspace_mode)` candidate space the lookahead
    picks from. Lookahead's p50 must be ≥ oracle p50 (ties allowed). If
    lookahead beats the oracle, either the oracle or the simulator is
    silently disagreeing on objective or candidate space."""
    name, episode, manifests, bundle, budget, warmness = _build_o2_inputs(scenario_index)
    oracle = run_small_n_oracle(
        episode, manifests, bundle, budget, warmness=warmness, max_workflows=4,
    )
    look_p50 = _result(
        mixed_lookahead(episode, manifests, bundle, warmness, budget),
        episode, manifests, bundle, warmness, budget,
    ).p50_resume_s()
    assert look_p50 >= oracle.oracle_p50_resume_s - 1e-9, (
        f"{name}: lookahead={look_p50:.4g} < oracle={oracle.oracle_p50_resume_s:.4g}"
    )


# ---------------------------------------------------------------------------
# Lookahead is a real heuristic (not a refactor) — distinguishes from no-op.
# ---------------------------------------------------------------------------


def test_lookahead_changes_plan_on_at_least_one_o2_cell():
    """Claim (non-trivial): on at least one O2 cell, mixed_lookahead's
    emitted plan differs from mixed_min_pressure's. Catches the failure
    mode where lookahead silently collapses to mixed (e.g., the score is
    always immediate_pressure and the residual term contributes nothing).
    """
    differences = []
    for i in range(len(O2_CELLS)):
        name, episode, manifests, bundle, budget, warmness = _build_o2_inputs(i)
        mixed = mixed_min_pressure(episode, manifests, bundle, warmness, budget)
        look = mixed_lookahead(episode, manifests, bundle, warmness, budget)
        for wf_id in mixed:
            m_choices = sorted((a.state_id, a.mode, a.dst_site) for a in mixed[wf_id])
            l_choices = sorted((a.state_id, a.mode, a.dst_site) for a in look[wf_id])
            if m_choices != l_choices:
                differences.append((name, wf_id))
                break
    assert differences, "lookahead never changes plans across O2 cells (likely a no-op refactor)"


def test_lookahead_makespan_no_worse_on_at_least_one_o2_cell():
    """Claim (load-bearing for the heuristic's STATED objective):
    `mixed_lookahead` optimizes a max-pressure proxy that corresponds to
    makespan. On at least one O2 cell, makespan must be no worse than
    `mixed_min_pressure`'s (preferably better). Otherwise the heuristic
    produces no benefit on its own optimization target."""
    margins = []
    look_better = 0
    for i in range(len(O2_CELLS)):
        name, episode, manifests, bundle, budget, warmness = _build_o2_inputs(i)
        mixed_makespan = _result(
            mixed_min_pressure(episode, manifests, bundle, warmness, budget),
            episode, manifests, bundle, warmness, budget,
        ).makespan_s
        look_makespan = _result(
            mixed_lookahead(episode, manifests, bundle, warmness, budget),
            episode, manifests, bundle, warmness, budget,
        ).makespan_s
        margins.append((name, mixed_makespan, look_makespan))
        if look_makespan <= mixed_makespan + 1e-9:
            look_better += 1
    assert look_better >= 1, f"lookahead makespan worse on every O2 cell: {margins}"


def test_lookahead_does_not_close_p50_gap_on_o2_cells():
    """Claim (NEGATIVE finding pinned as a regression sentinel): on the
    O2 diagnostic cells where mixed leaves substantial p50 headroom,
    lookahead does NOT close most of that gap. This is the load-bearing
    P1 finding — recorded so that:
      * a future refactor that silently 'fixes' the heuristic will fail
        this test (forcing review of whether the win is real or a
        candidate-space leak);
      * the negative result is verifiable from CI, not a doc-only claim.
    'Closes most' is defined as ≥50% of the gap. We assert the
    closure is BELOW 50% on every cell.
    """
    closures: list[tuple[str, float]] = []
    for i in range(len(O2_CELLS)):
        name, episode, manifests, bundle, budget, warmness = _build_o2_inputs(i)
        mixed_p50 = _result(
            mixed_min_pressure(episode, manifests, bundle, warmness, budget),
            episode, manifests, bundle, warmness, budget,
        ).p50_resume_s()
        look_p50 = _result(
            mixed_lookahead(episode, manifests, bundle, warmness, budget),
            episode, manifests, bundle, warmness, budget,
        ).p50_resume_s()
        oracle = run_small_n_oracle(
            episode, manifests, bundle, budget, warmness=warmness, max_workflows=4,
        )
        gap = max(mixed_p50 - oracle.oracle_p50_resume_s, 0.0)
        improvement = mixed_p50 - look_p50  # may be negative
        closure = improvement / gap if gap > 1e-9 else 0.0
        closures.append((name, closure))
    # Sentinel: max closure across all cells should be below 50%.
    # If a future change pushes this above 50%, the test fails — review the change
    # to confirm it is a legit improvement (then update this threshold).
    best_closure = max(c for _, c in closures)
    assert best_closure < 0.5, (
        f"lookahead now closes ≥50% of p50 gap on some cell: {closures}. "
        "If this is intentional, update the threshold and the policy docstring."
    )
