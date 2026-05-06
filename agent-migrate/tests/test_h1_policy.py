"""Tests for H1 (request_level_with_site_cache) — the fair baseline.

H1 places per_node like D1 but materializes once per (state, site) like D2/G1.
On every existing fixture (toy, g_demo, SWE_agent F2), every node's per_node
best_site is the same site, so H1 collapses to D2(tau=1) numerically. This is
a feature of those fixtures, not a bug in either policy. The constructed
divergence test below proves H1 != D2 fixtures exist."""
import json
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.adapters.multi_component import MultiComponentConfig, generate_to_file
from agent_migrate_agent.adapters.synthetic import write_jsonl
from agent_migrate_agent.policies import (
    Plan,
    run_g1_brute_force,
    run_request_level_no_reuse,
    run_request_level_with_site_cache,
    run_shared_state_aware,
)
from agent_migrate_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
TOY = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
G_DEMO = REPO / "examples" / "traces" / "g_demo_trace.jsonl"
SWE = REPO / "tests" / "fixtures" / "swe_agent_pilot_s_07.json"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _toy_manifest():
    return build_manifest(from_jsonl(str(TOY)))


def _g_demo_manifest(tmp_path: Path):
    out = tmp_path / "g_demo.jsonl"
    generate_to_file(MultiComponentConfig(), out)
    return build_manifest(from_jsonl(str(out)))


def _swe_manifest(tmp_path: Path):
    from agent_migrate_agent.adapters.swe_agent import swe_agent_to_trace
    trace = tmp_path / "swe.jsonl"
    swe_agent_to_trace(SWE, trace)
    return build_manifest(from_jsonl(str(trace)))


# ---- structural invariants ----

def test_h1_returns_plan_for_each_node():
    m = _toy_manifest()
    plan = run_request_level_with_site_cache(m, _bundle())
    assert {p.node_id for p in plan.placements} == set(m.nodes)
    assert plan.policy == "request_level_with_site_cache"


def test_h1_meta_is_empty():
    """H1 has no notion of components; meta must not surface a `components` key."""
    plan = run_request_level_with_site_cache(_toy_manifest(), _bundle())
    assert plan.meta == {}
    assert "components" not in plan.meta


def test_h1_materialization_reason_is_site_cache_reuse():
    plan = run_request_level_with_site_cache(_toy_manifest(), _bundle())
    assert all(m.reason == "site_cache_reuse" for m in plan.materializations)


def test_h1_placement_reason_is_min_cost():
    """H1 places per_node_min_cost; the placement reason should match D1's,
    not G1/G2's 'optimized' (which would mislead a debugger reading the plan)."""
    plan = run_request_level_with_site_cache(_toy_manifest(), _bundle())
    assert all(p.reason == "min_cost" for p in plan.placements)


def test_h1_materialization_count_always_one():
    """Per_site cache reuse: each (state, site) materializes exactly once."""
    plan = run_request_level_with_site_cache(_toy_manifest(), _bundle())
    assert all(m.materialization_count == 1 for m in plan.materializations)


# ---- H1 vs D1: bookkeeping is at_least_as_good ----

def test_h1_at_least_as_good_as_d1_on_toy():
    m = _toy_manifest()
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d1 = run_request_level_no_reuse(m, b)
    assert h1.total_cost_s() <= d1.total_cost_s() + 1e-9


def test_h1_placements_equal_d1_placements():
    """H1's placement decisions are identical to D1's per_node best_site
    selections; only the materialization bookkeeping differs."""
    m = _toy_manifest()
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d1 = run_request_level_no_reuse(m, b)
    h1_sites = {p.node_id: p.site for p in h1.placements}
    d1_sites = {p.node_id: p.site for p in d1.placements}
    assert h1_sites == d1_sites


# ---- H1 == D2(tau=1) on every existing fixture (the "collapse" finding) ----

def test_h1_equals_d2_on_toy():
    m = _toy_manifest()
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert h1.total_cost_s() == pytest.approx(d2.total_cost_s(), abs=1e-9)


def test_h1_equals_d2_on_g_demo(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert h1.total_cost_s() == pytest.approx(d2.total_cost_s(), abs=1e-9)


def test_h1_equals_d2_on_swe_agent(tmp_path: Path):
    m = _swe_manifest(tmp_path)
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert h1.total_cost_s() == pytest.approx(d2.total_cost_s(), abs=1e-9)


def test_h1_equals_g1_on_toy():
    m = _toy_manifest()
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    g1 = run_g1_brute_force(m, b)
    assert h1.total_cost_s() == pytest.approx(g1.total_cost_s(), abs=1e-9)


# ---- H1 < D2(tau=high) on g_demo: D2's fragmenting hurts ----

def test_h1_strictly_better_than_fragmenting_d2_on_g_demo(tmp_path: Path):
    """At tau=5000, D2 fragments and double_counts the cross_component shared
    state. H1 doesn't fragment (no notion of components) so it pays once."""
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2_frag = run_shared_state_aware(m, b, tau=5000)
    assert h1.total_cost_s() < d2_frag.total_cost_s()


# ---- mat_by_mat structural equivalence with D2 on toy ----

def test_h1_materialization_set_matches_d2_on_toy():
    m = _toy_manifest()
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    h1_keys = {(x.state_id, x.site) for x in h1.materializations}
    d2_keys = {(x.state_id, x.site) for x in d2.materializations}
    assert h1_keys == d2_keys


# ---- existence proof: H1 != D2 on a constructed fixture ----

def _h1_vs_d2_divergence_trace(tmp_path: Path) -> Path:
    """Two_node fixture exercising per_node_different_sites + same_component.

    Each node has a large *workspace* state anchored at a different site
    (1 GB; same_site cost = 0; cross_site = artifact_copy ~0.32s). The two
    share a small `prompt_shared` state (10 tokens, enough to put pair_weight
    above tau=1 so D2 merges them into one component) plus a small
    `ws_shared` workspace.

    Per_node best_sites: S1 -> phoenix (ws_x home), S2 -> seattle (ws_y home).
    Per_component best_site: phoenix (ws_y transferred once) vs seattle
    (ws_x transferred once + ws_shared transferred once); phoenix wins.

    H1 pays only the cross_site cost of the small shared states.
    D2 forces colocation and pays the LARGE ws_y transfer.
    """
    LARGE = 1_000_000_000   # 1 GB workspace
    SMALL = 100_000_000     # 100 MB workspace
    SHARED_TOKENS = 10      # > tau=1 so D2 merges S1+S2 into one component

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "h1_div"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n1", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "h1_div"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"description": "n2", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "h1_div"}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ws_x", "content_hash": "h", "layer": "workspace",
                     "lifetime": "private", "tokens": 0, "bytes": LARGE,
                     "producer_node_id": None, "home_site": "phoenix"},
         "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ws_y", "content_hash": "h", "layer": "workspace",
                     "lifetime": "private", "tokens": 0, "bytes": LARGE,
                     "producer_node_id": None, "home_site": "seattle"},
         "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ws_shared", "content_hash": "h",
                     "layer": "workspace", "lifetime": "shared",
                     "tokens": 0, "bytes": SMALL,
                     "producer_node_id": None, "home_site": "phoenix"},
         "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "prompt_shared", "content_hash": "h",
                     "layer": "prompt_context", "lifetime": "shared",
                     "tokens": SHARED_TOKENS, "bytes": None,
                     "producer_node_id": None, "home_site": "phoenix"},
         "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_x", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": 0}, "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_y", "content_hash": "h",
                     "consumer_node_id": "S2", "tokens": 0}, "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_shared", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": 0}, "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_shared", "content_hash": "h",
                     "consumer_node_id": "S2", "tokens": 0}, "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "prompt_shared", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": SHARED_TOKENS},
         "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "prompt_shared", "content_hash": "h",
                     "consumer_node_id": "S2", "tokens": SHARED_TOKENS},
         "reason": None},
    ]
    path = tmp_path / "h1_div.jsonl"
    write_jsonl(events, path)
    return path


def test_h1_diverges_from_d2_on_constructed_fixture(tmp_path: Path):
    """Existence proof: H1 != D2 fixtures exist, even though no current fixture
    triggers it. Direction here: H1 < D2 because D2's grouping forces a large
    workspace move that per_node placement avoids."""
    path = _h1_vs_d2_divergence_trace(tmp_path)
    m = build_manifest(from_jsonl(str(path)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert h1.total_cost_s() < d2.total_cost_s()


def test_h1_constructed_fixture_places_nodes_at_different_sites(tmp_path: Path):
    """Confirm the constructed fixture actually exercises the per_node_different-
    sites case. If this drifts (e.g., site config changes), the divergence
    test above will silently regress to equality."""
    path = _h1_vs_d2_divergence_trace(tmp_path)
    m = build_manifest(from_jsonl(str(path)))
    h1 = run_request_level_with_site_cache(m, _bundle())
    sites = {p.node_id: p.site for p in h1.placements}
    assert sites["S1"] == "phoenix"
    assert sites["S2"] == "seattle"


# ---- registry ----

def test_h1_registered_in_policies():
    from agent_migrate_agent.policies import POLICIES
    assert "request_level_with_site_cache" in POLICIES


def test_run_policy_dispatches_h1():
    from agent_migrate_agent.policies import run_policy
    plan = run_policy("request_level_with_site_cache", _toy_manifest(), _bundle())
    assert plan.policy == "request_level_with_site_cache"
