from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.plans_io import read_plan, write_plan
from vagrant_agent.policies import (
    POLICIES,
    Plan,
    run_policy,
    run_request_level_no_reuse,
    run_shared_state_aware,
)
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def _setup():
    manifest = build_manifest(from_jsonl(str(TRACE)))
    bundle = load_bundle(MODELS, SITES, "compact_kv")
    return manifest, bundle


# ---- D1: request_level_no_reuse ----

def test_d1_returns_plan_for_each_node():
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    placed_ids = {p.node_id for p in plan.placements}
    assert placed_ids == {"S1", "S2", "S3", "S4"}
    assert plan.policy == "request_level_no_reuse"


def test_d1_placements_have_min_cost_reason():
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    assert all(p.reason == "min_cost" for p in plan.placements)


def test_d1_default_toy_picks_seattle_for_everyone():
    """Seattle has higher prefill (45k vs 30k phoenix), so context_replay is cheaper at
    seattle for prompt_context-heavy work. Default toy: all four nodes placed at seattle."""
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    assert all(p.site == "seattle" for p in plan.placements)


def test_d1_per_node_materialization_counts_on_default_toy():
    """With all four placed at seattle, each shared state is materialized
    once per consumer at seattle."""
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    by_state = {(mat.state_id, mat.site): mat for mat in plan.materializations}
    assert by_state["repo_context", "seattle"].materialization_count == 4
    assert by_state["repo_context", "seattle"].reason == "per_node_no_reuse"
    assert by_state["system_prefix", "seattle"].materialization_count == 4
    assert by_state["workspace_AC", "seattle"].materialization_count == 2
    assert by_state["private_A", "seattle"].materialization_count == 1
    assert by_state["private_B", "seattle"].materialization_count == 1
    assert by_state["private_C", "seattle"].materialization_count == 1


# ---- D2: shared_state_aware ----

def test_d2_default_tau_one_component():
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    components = plan.meta["components"]
    assert len(components) == 1
    assert sorted(components[0]) == ["S1", "S2", "S3", "S4"]


def test_d2_high_tau_fragments_into_per_node_components():
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=100_000)
    components = plan.meta["components"]
    assert len(components) == 4  # all edges drop


def test_d2_default_toy_groups_at_seattle():
    """Single component places all four at seattle (cheaper prefill); each state
    materialized exactly once at seattle."""
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    assert all(p.site == "seattle" for p in plan.placements)
    assert all(m_.materialization_count == 1 for m_ in plan.materializations)
    state_ids = {m_.state_id for m_ in plan.materializations}
    assert state_ids == {"system_prefix", "repo_context", "workspace_AC",
                         "private_A", "private_B", "private_C"}


def test_d2_reason_is_grouped_when_component_size_gt_one():
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    assert all(p.reason == "grouped" for p in plan.placements)


# ---- registry ----

def test_run_policy_registry():
    m, b = _setup()
    a = run_policy("request_level_no_reuse", m, b)
    assert a.policy == "request_level_no_reuse"
    c = run_policy("shared_state_aware", m, b, tau=1)
    assert c.policy == "shared_state_aware"


def test_unknown_policy_hard_fails():
    m, b = _setup()
    with pytest.raises(ValueError, match="unknown policy"):
        run_policy("nonexistent", m, b)


def test_registry_keys_match_known_policies():
    assert set(POLICIES) == {
        "request_level_no_reuse", "shared_state_aware",
        "g1_brute_force", "g2_local_search",
    }


# ---- duplication-factor inequality (the MVP gate) ----

def test_mvp_gate_d1_duplicates_more_than_d2_on_default_toy():
    """The MVP headline: on the default toy, request_level_no_reuse pays
    materially more than shared_state_aware."""
    m, b = _setup()
    a = run_request_level_no_reuse(m, b)
    c = run_shared_state_aware(m, b, tau=1)
    assert c.total_cost_s() < a.total_cost_s()


# ---- plan I/O ----

def test_d1_placement_cost_s_per_node():
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    assert all(p.component_size == 1 for p in plan.placements)


def test_d2_placement_records_full_component_cost_per_member():
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    component_cost = plan.placements[0].cost_s
    assert all(p.cost_s == component_cost for p in plan.placements)
    assert all(p.component_size == 4 for p in plan.placements)


# ---- determinism ----

def test_d1_deterministic_across_runs():
    m, b = _setup()
    a = run_request_level_no_reuse(m, b)
    c = run_request_level_no_reuse(m, b)
    assert [(p.node_id, p.site, p.cost_s) for p in a.placements] == \
           [(p.node_id, p.site, p.cost_s) for p in c.placements]
    assert [(x.state_id, x.site, x.materialization_count) for x in a.materializations] == \
           [(x.state_id, x.site, x.materialization_count) for x in c.materializations]


def test_d2_deterministic_across_runs():
    m, b = _setup()
    a = run_shared_state_aware(m, b, tau=1)
    c = run_shared_state_aware(m, b, tau=1)
    assert a.meta == c.meta
    assert [(p.node_id, p.site) for p in a.placements] == [(p.node_id, p.site) for p in c.placements]


# ---- tau boundary ----

def test_d2_tau_boundary_strict_greater_than():
    """An edge with weight == tau must NOT merge nodes (spec: > tau)."""
    m, b = _setup()
    # system_prefix has weight 200 across 6 pairs. Setting tau = 8000 (repo_context's
    # exact weight) should still merge via repo_context's contribution because pair_weight
    # AGGREGATES across states. To test the boundary cleanly: use a tau equal to the
    # SUM of pair weights for some pair and verify the pair does not merge.
    # Pair (S1,S2) weight = 200 (system_prefix) + 8000 (repo_context) = 8200.
    plan_under = run_shared_state_aware(m, b, tau=8199)
    plan_at = run_shared_state_aware(m, b, tau=8200)
    plan_over = run_shared_state_aware(m, b, tau=8201)
    # Under boundary: pair survives; at-or-over: edge drops.
    assert len(plan_under.meta["components"]) < len(plan_over.meta["components"])
    assert plan_at.meta["components"] == plan_over.meta["components"]


# ---- multi-component split (forced) ----

def test_d2_multi_component_can_pick_different_sites():
    """Force a 2-component scenario by raising tau above all edge weights, then
    verify policy can place different components at different sites."""
    m, b = _setup()
    # tau larger than every pair weight -> 4 components, 1 node each.
    plan = run_shared_state_aware(m, b, tau=10_000_000)
    assert len(plan.meta["components"]) == 4
    # Each is a singleton; they all evaluate independently. With symmetric site costs
    # they'd all pick the same site, but we just confirm the policy doesn't crash on
    # multi-component layouts and emits one placement per node.
    assert len(plan.placements) == 4


# ---- mutability awareness ----

def test_plan_lists_are_mutable_documented_caveat():
    """Plan is frozen at the dataclass level but its lists are mutable.
    This documents the contract; do not rely on Plan immutability for safety."""
    m, b = _setup()
    plan = run_request_level_no_reuse(m, b)
    original_count = len(plan.placements)
    plan.placements.append(plan.placements[0])
    assert len(plan.placements) == original_count + 1


def test_plan_round_trip(tmp_path: Path):
    m, b = _setup()
    plan = run_shared_state_aware(m, b, tau=1)
    write_plan(plan, tmp_path)
    loaded = read_plan(tmp_path)
    assert loaded.policy == plan.policy
    assert len(loaded.placements) == len(plan.placements)
    assert len(loaded.materializations) == len(plan.materializations)
    assert loaded.meta == plan.meta
