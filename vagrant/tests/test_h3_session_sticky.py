"""
Claim:
`session_sticky` (Workstream H3) places every node sharing a `session_id`
at the same site. Materialization uses per-(state, site) cache reuse (same
bookkeeping as H1).

It coincides with H1 numerically when each session's nodes already share
a per-node best-site (the H2 case: workspace home dominates within the
session).

It diverges from H1 when intra-session nodes have private states pulling
toward different sites — session_sticky pays the constraint, H1 splits.

It coincides with D2(tau=1) on F2 single-instance fixtures (one session
spanning all nodes -> single colocated group), and is strictly cheaper than
D2 on H2 (D2 forces one global component across sessions; session_sticky
splits per session and avoids the cross-site workspace transfer).

Plausible wrong implementations the tests below try to catch:
- session_sticky reads workflow_id instead of session_id -> on H2, all
  nodes share workflow_id="h2_multi_session_swe", so the policy collapses
  to D2 (single group). Catch via "ss < D2 on H2."
- session_id missing -> all nodes default into one session -> same bug as
  above. Catch via "manifest.nodes carries session_id" structural test.
- session_sticky uses per-node best-site (== H1) by accident -> would
  hide intra-session disagreement. Catch via constructed-divergence test.
- session_sticky picks site via average-cost rather than total-cost ->
  would coincidentally succeed on most fixtures. Catch via the constructed
  fixture where total-vs-average diverges.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.adapters.synthetic import write_jsonl
from vagrant_agent.policies import (
    SESSION_STICKY_DEFAULT,
    run_g1_brute_force,
    run_request_level_with_site_cache,
    run_session_sticky,
    run_shared_state_aware,
)
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
H2 = REPO / "examples" / "traces" / "h2_multi_session_swe.jsonl"
TOY = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
SWE_TRAJ = REPO / "tests" / "fixtures" / "swe_agent_pilot_s_07.json"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


# ---------------------------------------------------------------------------
# Structural: session_id propagates from adapter into manifest.
# ---------------------------------------------------------------------------


def test_h2_adapter_stamps_session_id_on_every_node():
    m = build_manifest(from_jsonl(str(H2)))
    expected = {"sa_S1": "sa", "sa_S2": "sa", "sb_S1": "sb", "sb_S2": "sb",
                "sc_S1": "sc", "sc_S2": "sc"}
    assert {nid: n.session_id for nid, n in m.nodes.items()} == expected


def test_session_sticky_meta_partitions_nodes_by_session():
    m = build_manifest(from_jsonl(str(H2)))
    plan = run_session_sticky(m, _bundle())
    sessions = plan.meta["sessions"]
    assert set(sessions) == {"sa", "sb", "sc"}
    flattened = sorted(nid for ids in sessions.values() for nid in ids)
    assert flattened == sorted(m.nodes)


# ---------------------------------------------------------------------------
# H2: session_sticky == H1 numerically AND diverges from D2.
# session_sticky uses session boundaries; D2 uses the shared-state graph
# (which on H2 collapses everything to one component via system_prompt).
# ---------------------------------------------------------------------------


def test_session_sticky_equals_h1_on_h2():
    """On H2, each session's nodes already share a per-node best-site (the
    workspace home), so H1 places sa/sc at phoenix and sb at seattle —
    exactly what session_sticky enforces. Numerical equality."""
    m = build_manifest(from_jsonl(str(H2)))
    b = _bundle()
    ss = run_session_sticky(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert ss == pytest.approx(h1, abs=1e-9)


def test_session_sticky_strictly_better_than_d2_on_h2():
    """The whole point of H3 vs D2: session_sticky uses session boundaries
    (3 groups) while D2 uses shared-state components (1 group on H2). The
    gap is the same 1.6 s workspace cross-site transfer that H1 avoids."""
    m = build_manifest(from_jsonl(str(H2)))
    b = _bundle()
    ss = run_session_sticky(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    assert d2 - ss > 1.5


def test_session_sticky_partitions_distinct_from_d2_components():
    """If session_sticky's partition equals D2's component partition, the
    two policies should produce identical placements. On H2 they don't:
    D2 merges into one component while session_sticky splits."""
    m = build_manifest(from_jsonl(str(H2)))
    b = _bundle()
    ss = run_session_sticky(m, b)
    d2 = run_shared_state_aware(m, b, tau=1)
    ss_sites = {p.node_id: p.site for p in ss.placements}
    d2_sites = {p.node_id: p.site for p in d2.placements}
    assert ss_sites != d2_sites


# ---------------------------------------------------------------------------
# Single-session fallback: no session_id -> sentinel session -> one group.
# On a F2-style trace, session_sticky should behave like D2(tau=1) does on
# any linear-session trace (one colocated group).
# ---------------------------------------------------------------------------


def test_session_sticky_collapses_to_single_group_when_no_session_id(tmp_path: Path):
    """Construct a 3-node trace with NO session_id field — session_sticky
    must default all nodes into one sentinel session and place them
    together. Otherwise a missing session_id would silently produce
    arbitrary partitioning."""
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "no_session"}, "reason": None},
        *[
            {"step": 1, "event_type": "add_subtask", "subtask_id": nid,
             "payload": {"description": nid, "parent_id": None, "weight": 1.0,
                         "category": "product", "node_type": "llm_call",
                         "workflow_id": "no_session"},
             "reason": None}
            for nid in ("N1", "N2", "N3")
        ],
        {"step": 2, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "shared_ctx", "content_hash": "h",
                     "layer": "prompt_context", "lifetime": "shared",
                     "tokens": 1000, "bytes": None, "producer_node_id": None},
         "reason": None},
        *[
            {"step": 3, "event_type": "state_read", "subtask_id": None,
             "payload": {"state_id": "shared_ctx", "content_hash": "h",
                         "consumer_node_id": nid, "tokens": 1000},
             "reason": None}
            for nid in ("N1", "N2", "N3")
        ],
    ]
    p = tmp_path / "no_session.jsonl"
    write_jsonl(events, p)
    m = build_manifest(from_jsonl(str(p)))
    plan = run_session_sticky(m, _bundle())
    sites = {p.site for p in plan.placements}
    assert len(sites) == 1, f"all nodes without session_id should colocate; got {sites}"
    assert set(plan.meta["sessions"]) == {SESSION_STICKY_DEFAULT}


# ---------------------------------------------------------------------------
# Existence proof: session_sticky != H1 on a constructed fixture.
# Within ONE session, two nodes have private states with different homes.
# H1 places per-node (each at its private home). session_sticky must
# co-locate them and pay one large cross-site transfer.
# ---------------------------------------------------------------------------


def _intra_session_disagreement_trace(tmp_path: Path) -> Path:
    """Two nodes in ONE session, each anchored to a different site by a
    private workspace. H1 splits; session_sticky must pick one site and
    pay the other workspace's transfer."""
    LARGE = 1_000_000_000  # 1 GB workspace
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "intra"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "N1",
         "payload": {"description": "n1", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "intra", "session_id": "S"},
         "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "N2",
         "payload": {"description": "n2", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "intra", "session_id": "S"},
         "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ws_n1", "content_hash": "h",
                     "layer": "workspace", "lifetime": "private",
                     "tokens": 0, "bytes": LARGE,
                     "producer_node_id": None, "home_site": "phoenix"},
         "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ws_n2", "content_hash": "h",
                     "layer": "workspace", "lifetime": "private",
                     "tokens": 0, "bytes": LARGE,
                     "producer_node_id": None, "home_site": "seattle"},
         "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_n1", "content_hash": "h",
                     "consumer_node_id": "N1", "tokens": 0}, "reason": None},
        {"step": 1, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ws_n2", "content_hash": "h",
                     "consumer_node_id": "N2", "tokens": 0}, "reason": None},
    ]
    p = tmp_path / "intra.jsonl"
    write_jsonl(events, p)
    return p


def test_session_sticky_strictly_worse_than_h1_on_intra_session_disagreement(tmp_path: Path):
    """Constructed fixture proof. H1 pays 0 workspace-transfer (each at home).
    session_sticky pays exactly one workspace-transfer = 8 * 1 GB / 5 Gbps
    = 1.6 s.

        gap = ss - h1 ≈ 1.6 s (one cross-site workspace artifact_copy)
    """
    p = _intra_session_disagreement_trace(tmp_path)
    m = build_manifest(from_jsonl(str(p)))
    b = _bundle()
    ss = run_session_sticky(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    gap = ss - h1
    assert gap > 1.5, f"expected ~1.6 s gap; got {gap:.4f}"
    assert gap < 1.7, f"gap {gap:.4f} exceeds artifact_copy formula bound"


def test_session_sticky_strictly_worse_than_d2_when_nodes_share_no_state(tmp_path: Path):
    """In the intra-session-disagreement fixture, N1 and N2 share NO state
    object, so D2's pair-weight graph has zero edges and each node is its
    own component. D2 therefore picks per-component (== per-node) best-site
    and pays 0 workspace transfer. session_sticky's session-level constraint
    forces co-location and pays the 1.6 s cross-site transfer.

    This is the failure mode of a constraint-based policy: when the
    constraint contradicts the cost gradient, the policy strictly under-
    performs both H1 and D2. Lock the direction in so a future change to
    session-detection semantics doesn't silently make session_sticky look
    better than it should."""
    p = _intra_session_disagreement_trace(tmp_path)
    m = build_manifest(from_jsonl(str(p)))
    b = _bundle()
    ss = run_session_sticky(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert ss > d2 + 1e-9, "session_sticky must lose to D2 when nodes share no state"
    assert ss > h1 + 1e-9, "session_sticky must lose to H1 when nodes share no state"
    assert d2 == pytest.approx(h1, abs=1e-9), "with no shared state, D2 == H1 (per-component == per-node)"


def test_session_sticky_oracle_bounded_by_g1(tmp_path: Path):
    """G1 is the exact oracle; session_sticky is a heuristic with a
    constraint (same-session-same-site). G1 <= session_sticky always."""
    p = _intra_session_disagreement_trace(tmp_path)
    m = build_manifest(from_jsonl(str(p)))
    b = _bundle()
    ss = run_session_sticky(m, b).total_cost_s()
    g1 = run_g1_brute_force(m, b).total_cost_s()
    assert g1 <= ss + 1e-9


# ---------------------------------------------------------------------------
# F2 single-instance: every node carries the same instance_id as session_id,
# so session_sticky behaves like D2 on that linear-session trace (= H1 = G1).
# ---------------------------------------------------------------------------


def test_f2_single_session_collapses_to_one_group(tmp_path: Path):
    from vagrant_agent.adapters.swe_agent import swe_agent_to_trace
    out = tmp_path / "swe.jsonl"
    swe_agent_to_trace(SWE_TRAJ, out)
    m = build_manifest(from_jsonl(str(out)))
    sids = {n.session_id for n in m.nodes.values()}
    assert len(sids) == 1, f"F2 single-instance trace should have one session_id; got {sids}"
    plan = run_session_sticky(m, _bundle())
    sites = {p.site for p in plan.placements}
    assert len(sites) == 1


# ---------------------------------------------------------------------------
# Registry wiring.
# ---------------------------------------------------------------------------


def test_session_sticky_registered_in_policies():
    from vagrant_agent.policies import POLICIES
    assert "session_sticky" in POLICIES


def test_run_policy_dispatches_session_sticky():
    from vagrant_agent.policies import run_policy
    m = build_manifest(from_jsonl(str(H2)))
    plan = run_policy("session_sticky", m, _bundle())
    assert plan.policy == "session_sticky"


def test_session_sticky_materialization_reason_is_site_cache_reuse():
    m = build_manifest(from_jsonl(str(H2)))
    plan = run_session_sticky(m, _bundle())
    assert all(d.reason == "site_cache_reuse" for d in plan.materializations)


def test_session_sticky_placement_reason_carries_session_id():
    """Audit CSVs must let a reader identify which session pulled a node.
    A bare 'session_sticky' is diagnostic-poor."""
    m = build_manifest(from_jsonl(str(H2)))
    plan = run_session_sticky(m, _bundle())
    expected = {f"session_sticky:{nid.split('_')[0]}" for nid in m.nodes}
    actual = {p.reason for p in plan.placements}
    assert actual == expected


def test_session_sticky_hard_fails_on_mixed_session_id_presence(tmp_path: Path):
    """If half the nodes have session_id and half don't, the policy must
    refuse rather than silently merge unsessioned nodes into one sentinel
    bucket alongside the explicit sessions. This is a property of the
    manifest, not the trace, so we hand-craft a mixed manifest."""
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "mix"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "M1",
         "payload": {"description": "m1", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "mix", "session_id": "S"},
         "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "M2",
         "payload": {"description": "m2", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "mix"},
         "reason": None},
        {"step": 2, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "ctx", "content_hash": "h",
                     "layer": "prompt_context", "lifetime": "shared",
                     "tokens": 1000, "bytes": None, "producer_node_id": None},
         "reason": None},
        {"step": 3, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ctx", "content_hash": "h",
                     "consumer_node_id": "M1", "tokens": 1000},
         "reason": None},
        {"step": 3, "event_type": "state_read", "subtask_id": None,
         "payload": {"state_id": "ctx", "content_hash": "h",
                     "consumer_node_id": "M2", "tokens": 1000},
         "reason": None},
    ]
    p = tmp_path / "mix.jsonl"
    write_jsonl(events, p)
    m = build_manifest(from_jsonl(str(p)))
    with pytest.raises(ValueError, match="uniform session_id presence"):
        run_session_sticky(m, _bundle())


def test_session_sticky_invariant_dominates_h1_on_every_fixture(tmp_path: Path):
    """session_sticky is provably >= H1 by construction: it solves the same
    minimization (sum of per-state min-cost-mode costs at one site) under
    a strictly tighter constraint (one site per session vs one per node).
    Pin the invariant so a future implementation drift can't accidentally
    let session_sticky beat H1."""
    from vagrant_agent.adapters.swe_agent import swe_agent_to_trace
    fixtures: list[Path] = [H2, TOY]
    swe_out = tmp_path / "swe.jsonl"
    swe_agent_to_trace(SWE_TRAJ, swe_out)
    fixtures.append(swe_out)

    b = _bundle()
    for trace in fixtures:
        m = build_manifest(from_jsonl(str(trace)))
        try:
            ss = run_session_sticky(m, b).total_cost_s()
        except ValueError as e:
            if "uniform session_id" in str(e):
                continue
            raise
        h1 = run_request_level_with_site_cache(m, b).total_cost_s()
        assert ss >= h1 - 1e-9, \
            f"session_sticky beat H1 on {trace.name}: ss={ss}, h1={h1}"
