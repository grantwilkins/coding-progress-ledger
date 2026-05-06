"""Tests for G1 (brute_force oracle) and G2 (local search from D1).

Demonstrates G's value relative to D2 on a multi_component fixture, and
guards against G1/G2 ever doing WORSE than D2 (the optimizer is at_least-
as_good as the heuristics)."""
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.adapters.multi_component import MultiComponentConfig, generate_to_file
from agent_migrate_agent.policies import (
    G1_MAX_ENUMERATIONS,
    run_g1_brute_force,
    run_g2_local_search,
    run_request_level_no_reuse,
    run_shared_state_aware,
)
from agent_migrate_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
TOY = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
G_DEMO = REPO / "examples" / "traces" / "g_demo_trace.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


# ---- G demo fixture: regenerate to ensure it tracks generator changes ----

def _g_demo_manifest(tmp_path: Path):
    out = tmp_path / "g_demo.jsonl"
    generate_to_file(MultiComponentConfig(), out)
    return build_manifest(from_jsonl(str(out)))


# ---- G1 correctness ----

def test_g1_returns_plan_for_each_node(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    plan = run_g1_brute_force(m, _bundle())
    assert {p.node_id for p in plan.placements} == set(m.nodes)
    assert plan.policy == "g1_brute_force"


def test_g1_enumeration_count(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    bundle = _bundle()
    plan = run_g1_brute_force(m, bundle)
    expected = len(bundle.sites) ** len(m.nodes)
    assert plan.meta["enumerated"] == expected


def test_g1_at_least_as_good_as_d2_on_g_demo(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    g1 = run_g1_brute_force(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert g1.total_cost_s() <= d2.total_cost_s() + 1e-9


def test_g1_strictly_beats_d2_at_fragmenting_tau(tmp_path: Path):
    """At tau=5000, D2 fragments into two components and pays per_component
    bookkeeping for the cross_component shared state. G1 sees the global
    structure and avoids the duplication."""
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    g1 = run_g1_brute_force(m, b)
    d2_frag = run_shared_state_aware(m, b, tau=5000)
    assert g1.total_cost_s() < d2_frag.total_cost_s()


def test_g1_at_least_as_good_as_d1(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    g1 = run_g1_brute_force(m, b)
    d1 = run_request_level_no_reuse(m, b)
    assert g1.total_cost_s() <= d1.total_cost_s()


def test_g1_enumeration_cap_hard_fails(tmp_path: Path):
    """Construct a manifest large enough that K^N > G1_MAX_ENUMERATIONS;
    G1 must refuse rather than silently fall back."""
    from math import ceil, log
    from agent_migrate_agent.adapters.synthetic import write_jsonl

    bundle = _bundle()
    k = len(bundle.sites)
    n = ceil(log(G1_MAX_ENUMERATIONS, k)) + 1  # smallest N with k^N > cap
    events = [{"step": 0, "event_type": "init", "subtask_id": None,
               "payload": {"root_task": "big"}, "reason": None}]
    for i in range(n):
        events.append({
            "step": 1, "event_type": "add_subtask", "subtask_id": f"S{i + 1}",
            "payload": {"description": f"n{i}", "parent_id": None, "weight": 1.0,
                        "category": "product", "node_type": "llm_call"},
            "reason": None,
        })
    path = tmp_path / "big.jsonl"
    write_jsonl(events, path)
    manifest = build_manifest(from_jsonl(str(path)))
    with pytest.raises(ValueError, match="placement space"):
        run_g1_brute_force(manifest, bundle)


# ---- G2 correctness ----

def test_g2_returns_plan(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    plan = run_g2_local_search(m, _bundle())
    assert plan.policy == "g2_local_search"
    assert {p.node_id for p in plan.placements} == set(m.nodes)


def test_g2_at_least_as_good_as_d1(tmp_path: Path):
    """G2 is seeded from D1; local search only accepts strict improvements,
    so it can never do worse."""
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    g2 = run_g2_local_search(m, b)
    d1 = run_request_level_no_reuse(m, b)
    assert g2.total_cost_s() <= d1.total_cost_s() + 1e-9


def test_g2_terminates(tmp_path: Path):
    m = _g_demo_manifest(tmp_path)
    plan = run_g2_local_search(m, _bundle(), max_iterations=10)
    assert plan.meta["iterations"] <= 10


def test_g2_matches_g1_on_small_instance(tmp_path: Path):
    """For convex / single_min instances (no local optima), G2 should reach G1's
    optimum. The G demo fixture has 4 nodes × 2 sites = 16 placements."""
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    g1 = run_g1_brute_force(m, b)
    g2 = run_g2_local_search(m, b)
    assert g2.total_cost_s() == pytest.approx(g1.total_cost_s(), abs=1e-9)


# ---- toy regression: G must not regress D1/D2 on the existing toy ----

def test_g1_on_toy_at_least_as_good_as_d2():
    m = build_manifest(from_jsonl(str(TOY)))
    b = _bundle()
    g1 = run_g1_brute_force(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert g1.total_cost_s() <= d2.total_cost_s() + 1e-9


def test_g2_on_toy_at_least_as_good_as_d2():
    m = build_manifest(from_jsonl(str(TOY)))
    b = _bundle()
    g2 = run_g2_local_search(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    assert g2.total_cost_s() <= d2.total_cost_s() + 1e-9


# ---- registry ----

def test_run_policy_registry_includes_g(tmp_path: Path):
    from agent_migrate_agent.policies import run_policy
    m = _g_demo_manifest(tmp_path)
    b = _bundle()
    p1 = run_policy("g1_brute_force", m, b)
    p2 = run_policy("g2_local_search", m, b)
    assert p1.policy == "g1_brute_force"
    assert p2.policy == "g2_local_search"


def test_g_demo_committed_matches_generator(tmp_path: Path):
    """The committed g_demo trace must match the generator output byte_for_byte
    so changes to MultiComponentConfig defaults can't silently drift."""
    expected = tmp_path / "expected.jsonl"
    generate_to_file(MultiComponentConfig(), expected)
    assert G_DEMO.read_bytes() == expected.read_bytes()


def test_invalid_home_site_hard_fails_at_policy_entry(tmp_path: Path):
    """A state.home_site referencing a site not in the bundle must surface the
    typo at policy entry, not deep inside choose_min_cost_mode mid_enumeration."""
    from agent_migrate_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 100, "bytes": None,
                     "producer_node_id": None, "home_site": "TYPO_NOT_A_SITE"},
         "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": 100}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    m = build_manifest(from_jsonl(str(path)))
    b = _bundle()
    for runner in (run_request_level_no_reuse,
                   lambda mm, bb: run_shared_state_aware(mm, bb, tau=1),
                   run_g1_brute_force,
                   run_g2_local_search):
        with pytest.raises(ValueError, match="not in bundle.sites"):
            runner(m, b)


def test_g_demo_fixture_has_2_components_at_tau_5000(tmp_path: Path):
    """Sanity: the G demo trace must produce 2 components at tau=5000 to make
    G1's value visible. If this drifts (because someone tunes the synthetic
    parameters), the G test suite will fail informatively."""
    m = _g_demo_manifest(tmp_path)
    plan = run_shared_state_aware(m, _bundle(), tau=5000)
    assert len(plan.meta["components"]) == 2
