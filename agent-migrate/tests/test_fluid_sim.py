"""
Claim:
K4's fluid simulator advances time event_by_event under proportional
fair_share resource splitting. Per_action wallclock under infinite
capacity equals its closed_form lower bound (costs.materialize_cost);
two actions sharing one resource finish in 2x the lower bound;
all_warm episodes execute zero actions (warmness short_circuit).

K4 is the ONLY module that mutates a warmness map. The simulator
returns a NEW WarmnessMap reflecting all materializations performed
during the episode. KV_memory pressure forces LRU eviction.

Plausible wrong implementations the tests below catch:
- a simulator that finishes all actions in zero time -> wall_clock
  parity test trips.
- proportional_share computed once_and_frozen across the episode
  (instead of recomputed per event horizon) -> the 2_actions_on_1-link
  test trips.
- KV_memory capacity ignored -> the LRU_eviction test trips.
- nondeterministic action ordering -> the determinism_under_seed test trips.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from agent_migrate_agent.episode import MobilityEpisode, Workflow
from agent_migrate_agent.fluid_sim import (
    ActionTrace,
    NETWORK,
    NONE,
    PREFILL,
    WORKSPACE,
    ReconstitutionAction,
    simulate_fluid,
)
from agent_migrate_agent.manifest import ServingGroupManifest, StateObject, WorkNode
from agent_migrate_agent.profiles import load_bundle
from agent_migrate_agent.resources import ResourceBudget
from agent_migrate_agent.warmness import WarmnessMap

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES_2 = REPO / "configs" / "sites_2site.yaml"
SITES_3 = REPO / "configs" / "sites_3site.yaml"


def _bundle_2site():
    return load_bundle(MODELS, SITES_2, "compact_kv")


def _bundle_3site():
    return load_bundle(MODELS, SITES_3, "compact_kv")


# ---------------------------------------------------------------------------
# Test fixture builders
# ---------------------------------------------------------------------------


def _make_node(node_id: str, required: list[str], wf: str = "wf") -> WorkNode:
    return WorkNode(
        node_id=node_id, node_type="llm_call",
        parent_node_id=None, workflow_id=wf, label=None, status="complete",
        required_state=list(required), produced_state=[],
    )


def _make_manifest(nodes: dict[str, WorkNode],
                   states: dict[str, StateObject],
                   wf: str = "wf") -> ServingGroupManifest:
    return ServingGroupManifest(
        workflow_id=wf, root_task="test",
        nodes=nodes, state_objects=states, edges=[],
    )


def _single_state_manifest(state_id: str = "ws", tokens: int = 0,
                           bytes_: int | None = 1_000_000_000,
                           layer: str = "workspace",
                           lifetime: str = "private") -> ServingGroupManifest:
    state = StateObject(
        state_id=state_id, content_hash=f"h_{state_id}", layer=layer,
        lifetime=lifetime, tokens=tokens, bytes=bytes_,
    )
    node = _make_node(f"n_{state_id}", [state_id])
    return _make_manifest({node.node_id: node}, {state.state_id: state})


def _single_workflow_episode(workflow_id: str = "wf", src: str = "phoenix",
                             dst: str = "seattle") -> MobilityEpisode:
    return MobilityEpisode(
        episode_id="ep_test",
        source_sites=(src,),
        destination_sites=(dst,),
        workflows=(Workflow(workflow_id=workflow_id, manifest_path="<inline>",
                            src_site=src),),
    )


# ---------------------------------------------------------------------------
# Gate 1: under infinite capacity, single action finishes at lower bound.
# ---------------------------------------------------------------------------


def test_single_action_infinite_capacity_matches_lower_bound():
    bundle = _bundle_2site()
    m = _single_state_manifest(layer="workspace", bytes_=1_000_000_000)
    ep = _single_workflow_episode()
    plan = {"wf": [ReconstitutionAction(
        workflow_id="wf", state_id="ws", mode="artifact_copy",
        src_site="phoenix", dst_site="seattle",
    )]}
    budget = ResourceBudget.infinite(["phoenix", "seattle"])
    # Override only the link bps so we have a finite, controllable rate.
    # Use 5 Gbps to match the canonical sites_2site config.
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 5e9},
        prefill_tok_s_per_site=budget.prefill_tok_s_per_site,
        workspace_hydrate_bps_per_site=budget.workspace_hydrate_bps_per_site,
        kv_memory_bytes_per_site=budget.kv_memory_bytes_per_site,
    )
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, WarmnessMap.empty(), budget)
    # 8 * 1e9 / 5e9 = 1.6 s
    assert len(result.actions) == 1
    assert result.actions[0].finished_s == pytest.approx(1.6, rel=1e-6)
    assert result.makespan_s == pytest.approx(1.6, rel=1e-6)
    assert result.actions[0].bottleneck == NETWORK


# ---------------------------------------------------------------------------
# Gate 2: two actions sharing one link → 2x slowdown.
# ---------------------------------------------------------------------------


def test_two_actions_one_link_slowdown_factor_2():
    bundle = _bundle_2site()
    # Two parallel workflows, each 1 GB workspace artifact_copy phoenix->seattle.
    m_a = _single_state_manifest(state_id="ws_a")
    m_b = _single_state_manifest(state_id="ws_b")
    ep = MobilityEpisode(
        episode_id="ep2",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(
            Workflow(workflow_id="a", manifest_path="<inline>", src_site="phoenix"),
            Workflow(workflow_id="b", manifest_path="<inline>", src_site="phoenix"),
        ),
    )
    plan = {
        "a": [ReconstitutionAction("a", "ws_a", "artifact_copy", "phoenix", "seattle")],
        "b": [ReconstitutionAction("b", "ws_b", "artifact_copy", "phoenix", "seattle")],
    }
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 5e9},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(
        ep, {"a": m_a, "b": m_b}, plan, bundle, WarmnessMap.empty(), budget,
    )
    # 8 * 1e9 / (5e9 / 2) = 3.2 s for each action sharing the link
    assert len(result.actions) == 2
    for a in result.actions:
        assert a.finished_s == pytest.approx(3.2, rel=1e-6)
    assert result.makespan_s == pytest.approx(3.2, rel=1e-6)


# ---------------------------------------------------------------------------
# Gate 3: all_warm episode → no actions execute (warmness short_circuit
# means reconstitution_cost = zero, but K4 only sees the action list it's
# given. If we give it no actions, makespan = 0 trivially. If we DO give it
# actions but they all hit warm caches, the action's remaining bytes are
# zero across all axes -> per_action_rate returns (inf, NONE) -> finishes
# in 0 time at the next event horizon. Test both.)
# ---------------------------------------------------------------------------


def test_empty_plan_zero_makespan():
    """Workflows with no actions in their plan finish at trigger_t_s."""
    bundle = _bundle_2site()
    m = _single_state_manifest()
    ep = _single_workflow_episode()
    plan = {"wf": []}
    budget = ResourceBudget.infinite(["phoenix", "seattle"])
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, WarmnessMap.empty(), budget)
    assert result.makespan_s == 0.0
    assert len(result.actions) == 0
    assert result.per_workflow_finish_s == {"wf": 0.0}


def test_all_warm_actions_finish_in_zero_time():
    """An action whose state is already warm at dst short_circuits
    reconstitution_cost to zero -> all remaining[axis] = 0 -> action
    finishes at the next event horizon (which arrives in 0 wallclock)."""
    bundle = _bundle_2site()
    m = _single_state_manifest(layer="workspace", bytes_=1_000_000_000)
    ep = _single_workflow_episode()
    plan = {"wf": [ReconstitutionAction(
        workflow_id="wf", state_id="ws", mode="artifact_copy",
        src_site="phoenix", dst_site="seattle",
    )]}
    warm = WarmnessMap.from_dict({"ws": ["seattle"]})
    budget = ResourceBudget.infinite(["phoenix", "seattle"])
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, warm, budget)
    assert result.makespan_s == 0.0
    assert len(result.actions) == 1
    assert result.actions[0].bottleneck == NONE


# ---------------------------------------------------------------------------
# Determinism: same input -> same output (action order, bottleneck attribution).
# ---------------------------------------------------------------------------


def test_deterministic_two_runs_identical():
    bundle = _bundle_2site()
    m_a = _single_state_manifest(state_id="ws_a")
    m_b = _single_state_manifest(state_id="ws_b")
    ep = MobilityEpisode(
        episode_id="ep_det", source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(
            Workflow("a", "<inline>", src_site="phoenix"),
            Workflow("b", "<inline>", src_site="phoenix"),
        ),
    )
    plan = {
        "a": [ReconstitutionAction("a", "ws_a", "artifact_copy", "phoenix", "seattle")],
        "b": [ReconstitutionAction("b", "ws_b", "artifact_copy", "phoenix", "seattle")],
    }
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 5e9},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    r1 = simulate_fluid(ep, {"a": m_a, "b": m_b}, plan, bundle, WarmnessMap.empty(), budget)
    r2 = simulate_fluid(ep, {"a": m_a, "b": m_b}, plan, bundle, WarmnessMap.empty(), budget)
    assert r1.makespan_s == r2.makespan_s
    assert r1.actions == r2.actions


# ---------------------------------------------------------------------------
# Warmness mutation: K4 produces a final warmness reflecting all
# successfully_completed reconstitutions.
# ---------------------------------------------------------------------------


def test_final_warmness_reflects_completed_reconstitutions():
    bundle = _bundle_2site()
    m = _single_state_manifest()
    ep = _single_workflow_episode()
    plan = {"wf": [ReconstitutionAction(
        workflow_id="wf", state_id="ws", mode="artifact_copy",
        src_site="phoenix", dst_site="seattle",
    )]}
    budget = ResourceBudget.infinite(["phoenix", "seattle"])
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 5e9},
        prefill_tok_s_per_site=budget.prefill_tok_s_per_site,
        workspace_hydrate_bps_per_site=budget.workspace_hydrate_bps_per_site,
        kv_memory_bytes_per_site=budget.kv_memory_bytes_per_site,
    )
    initial = WarmnessMap.empty()
    assert not initial.is_warm("ws", "seattle")
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, initial, budget)
    assert result.final_warmness.is_warm("ws", "seattle")
    # Original warmness unchanged (frozen dataclass).
    assert not initial.is_warm("ws", "seattle")


# ---------------------------------------------------------------------------
# Resource conservation: sum of network_bytes consumed across actions
# does not exceed link capacity * makespan.
# ---------------------------------------------------------------------------


def test_resource_conservation_network():
    """Total bits transferred across the link <= link_bps * makespan."""
    bundle = _bundle_2site()
    m_a = _single_state_manifest(state_id="ws_a", bytes_=500_000_000)
    m_b = _single_state_manifest(state_id="ws_b", bytes_=500_000_000)
    ep = MobilityEpisode(
        episode_id="ep_cons", source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(
            Workflow("a", "<inline>", src_site="phoenix"),
            Workflow("b", "<inline>", src_site="phoenix"),
        ),
    )
    plan = {
        "a": [ReconstitutionAction("a", "ws_a", "artifact_copy", "phoenix", "seattle")],
        "b": [ReconstitutionAction("b", "ws_b", "artifact_copy", "phoenix", "seattle")],
    }
    LINK = 5e9
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): LINK},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(
        ep, {"a": m_a, "b": m_b}, plan, bundle, WarmnessMap.empty(), budget,
    )
    total_bits = 8 * (500_000_000 + 500_000_000)
    capacity_bits = LINK * result.makespan_s
    # Conservation: with 100% utilization, sum = capacity * makespan.
    assert total_bits == pytest.approx(capacity_bits, rel=1e-6)


# ---------------------------------------------------------------------------
# Sequential workflow: actions in one workflow run sequentially, not in parallel.
# ---------------------------------------------------------------------------


def test_workflow_actions_run_sequentially():
    """Workflow w has 2 sequential actions; each must finish before the
    next starts. Total time = sum, not max."""
    bundle = _bundle_2site()
    m = _make_manifest(
        nodes={"n1": _make_node("n1", ["s1"], wf="w")},
        states={
            "s1": StateObject("s1", "h1", "workspace", "private", 0, 1_000_000_000),
            "s2": StateObject("s2", "h2", "workspace", "private", 0, 1_000_000_000),
        },
        wf="w",
    )
    ep = _single_workflow_episode("w")
    plan = {"w": [
        ReconstitutionAction("w", "s1", "artifact_copy", "phoenix", "seattle"),
        ReconstitutionAction("w", "s2", "artifact_copy", "phoenix", "seattle"),
    ]}
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): 5e9},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(ep, {"w": m}, plan, bundle, WarmnessMap.empty(), budget)
    # 1.6 s for s1, then 1.6 s for s2, sequentially -> 3.2 s
    assert result.makespan_s == pytest.approx(3.2, rel=1e-6)
    assert len(result.actions) == 2
    assert result.actions[0].finished_s < result.actions[1].started_s + 1e-9


# ---------------------------------------------------------------------------
# Bottleneck attribution: an action limited by prefill (replay) reports
# PREFILL as the bottleneck even when network capacity is ample.
# ---------------------------------------------------------------------------


def test_bottleneck_is_prefill_under_finite_prefill():
    bundle = _bundle_2site()
    m = _make_manifest(
        nodes={"n1": _make_node("n1", ["sx"])},
        states={"sx": StateObject("sx", "h", "prompt_context", "shared",
                                  tokens=10_000, bytes=None)},
    )
    ep = _single_workflow_episode()
    plan = {"wf": [ReconstitutionAction(
        "wf", "sx", "context_replay", "phoenix", "seattle",
    )]}
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): math.inf},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": 1000.0},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, WarmnessMap.empty(), budget)
    assert result.actions[0].bottleneck == PREFILL
    # 10000 tokens / 1000 tok_s = 10 s
    assert result.actions[0].finished_s == pytest.approx(10.0, rel=1e-6)


def test_workspace_hydrate_fluid_capacity_uses_bytes_per_second():
    """1 GB workspace hydrate at 1 GB/s consumes 1 second in K4."""
    bundle = _bundle_3site()
    m = _single_state_manifest(layer="workspace", bytes_=1_000_000_000)
    ep = MobilityEpisode(
        episode_id="ep_hydrate",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(Workflow("wf", "<inline>", src_site="phoenix"),),
    )
    plan = {"wf": [ReconstitutionAction(
        "wf", "ws", "workspace_hydrate", "seattle", "seattle",
    )]}
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): math.inf},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": 1_000_000_000},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, WarmnessMap.empty(), budget)
    assert result.makespan_s == pytest.approx(1.0, rel=1e-6)
    assert result.actions[0].bottleneck == WORKSPACE


def test_concurrent_shared_state_materializes_once_per_destination():
    """Two workflows needing the same cold state at one site should coalesce.

    Without in_flight per-(state, site) dedup, both replay actions would share
    the 1000 tok/s prefill pool and finish at 20s. Correct L1 materialization
    pays the 10k_token replay once, then the second workflow hits warmness at
    the same timestamp.
    """
    bundle = _bundle_2site()
    shared = StateObject("shared_prompt", "same_hash", "prompt_context", "shared",
                         tokens=10_000, bytes=None)
    m_a = _make_manifest(
        nodes={"n": _make_node("n", ["shared_prompt"], wf="a")},
        states={"shared_prompt": shared},
        wf="a",
    )
    m_b = _make_manifest(
        nodes={"n": _make_node("n", ["shared_prompt"], wf="b")},
        states={"shared_prompt": shared},
        wf="b",
    )
    ep = MobilityEpisode(
        episode_id="ep_shared",
        source_sites=("phoenix",),
        destination_sites=("seattle",),
        workflows=(
            Workflow("a", "<inline>", src_site="phoenix"),
            Workflow("b", "<inline>", src_site="phoenix"),
        ),
    )
    plan = {
        "a": [ReconstitutionAction("a", "shared_prompt", "context_replay", "phoenix", "seattle")],
        "b": [ReconstitutionAction("b", "shared_prompt", "context_replay", "phoenix", "seattle")],
    }
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): math.inf},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": 1000.0},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": math.inf},
    )
    result = simulate_fluid(
        ep, {"a": m_a, "b": m_b}, plan, bundle, WarmnessMap.empty(), budget,
    )
    assert result.makespan_s == pytest.approx(10.0, rel=1e-6)
    assert sorted(result.per_workflow_finish_s.values()) == pytest.approx([10.0, 10.0])


# ---------------------------------------------------------------------------
# KV_memory pressure: when a finishing action would exceed dst KV cap,
# LRU eviction triggers and warmness loses its oldest entry at dst.
# ---------------------------------------------------------------------------


def test_kv_memory_eviction_triggers_when_capacity_exceeded():
    bundle = _bundle_2site()
    # Two prompt_context states; each occupies tokens * kv_bytes_per_token.
    # Set kv_cap so that only ONE fits at dst.
    kv_per_token = bundle.model.kv_bytes_per_token  # 70656
    state_tokens = 10_000
    one_state_bytes = state_tokens * kv_per_token
    kv_cap = int(one_state_bytes * 1.5)  # only 1 fits

    m = _make_manifest(
        nodes={
            "n1": _make_node("n1", ["sx"]),
            "n2": _make_node("n2", ["sy"]),
        },
        states={
            "sx": StateObject("sx", "h", "prompt_context", "shared",
                              tokens=state_tokens, bytes=None),
            "sy": StateObject("sy", "h", "prompt_context", "shared",
                              tokens=state_tokens, bytes=None),
        },
    )
    ep = _single_workflow_episode()
    plan = {"wf": [
        ReconstitutionAction("wf", "sx", "context_replay", "phoenix", "seattle"),
        ReconstitutionAction("wf", "sy", "context_replay", "phoenix", "seattle"),
    ]}
    budget = ResourceBudget(
        network_bps_per_link={("phoenix", "seattle"): math.inf},
        prefill_tok_s_per_site={"phoenix": math.inf, "seattle": math.inf},
        workspace_hydrate_bps_per_site={"phoenix": math.inf, "seattle": math.inf},
        kv_memory_bytes_per_site={"phoenix": math.inf, "seattle": kv_cap},
    )
    result = simulate_fluid(ep, {"wf": m}, plan, bundle, WarmnessMap.empty(), budget)
    # Both finished, but the oldest (sx) should be evicted from final warmness.
    assert result.final_warmness.is_warm("sy", "seattle")
    assert not result.final_warmness.is_warm("sx", "seattle"), \
        "LRU should have evicted the older entry under capacity pressure"
