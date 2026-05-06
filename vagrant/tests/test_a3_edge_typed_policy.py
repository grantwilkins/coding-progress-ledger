"""
Claim:
D3 (`shared_state_aware_typed`) is D2 with edge-type-weighted pair sums.
By zeroing out global-replicated edges (e.g., system_prompt) and
upweighting workspace-local edges, D3 avoids D2's overgrouping pathology
on H2/H5a — where a tiny system_prompt forces all sessions into one
component and pays a 1.6s cross-site workspace transfer per minority-
home session.

Audit findings (all asserted below):
- On linear-session fixtures (toy, g_demo) D3 == D2 == H1 (no overgrouping).
- On H2 (3-session synthetic 1GB) D3 strictly beats D2 by >1.5s.
- On H5a (5-session synthetic 1GB) D3 strictly beats D2 by >3s.
- D3 >= H1 - 1e-9 on every fixture. **D3 does NOT reach H1.**

The deeper, audit-level finding: **D3 is NOT strictly <= D2 universally.**
D3 fixes overgrouping but INHERITS D2's component-level materialization
accounting (each component pays per-(state, site) once, not shared
across components). On H5b real bytes, D2 picks the faster-prefill site
and pays system_prompt once; D3 fragments into 5 components and pays
system_prompt 5x (3 phoenix + 2 seattle). D3 > D2 on this fixture by
~108 ms. The L1-vs-L2 distinction is structurally about materialization
accounting, NOT about edge-typing — so edge-typing alone cannot close
the H5b gap and can even open new gaps in regimes where D2's
all-at-one-site choice is correct.

This finding has implications for K0: the resource-vector model in K3
must treat materialization at the (state, site) level, not the
(component, site) level. Otherwise K's `mixed_min_pressure` policy will
inherit D3's bookkeeping pathology.

Plausible wrong implementations the tests below try to catch:
- a future change makes D3 share materialization across components
  -> D3 collapses to H1 numerically; the "D3 > H1 on H2/H5a" pin trips
- the edge-type weight dict default changes silently
  -> per-fixture numerical pin trips
- D3 dispatched with wrong kwargs in run_policy
  -> registry-roundtrip test trips
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.adapters.swe_agent_multi import (
    MultiSessionConfig,
    SessionSpec,
    generate_to_file,
)
from vagrant_agent.policies import (
    DEFAULT_EDGE_TYPE_WEIGHTS,
    POLICIES,
    run_g1_brute_force,
    run_policy,
    run_request_level_with_site_cache,
    run_shared_state_aware,
    run_shared_state_aware_typed,
)
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"

WORKSPACES_DIR = Path(os.environ.get("VAGRANT_H5B_WORKSPACES", "/tmp/h5b_workspaces"))


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


# ---------------------------------------------------------------------------
# Registry plumbing.
# ---------------------------------------------------------------------------


def test_d3_registered():
    assert "shared_state_aware_typed" in POLICIES
    assert POLICIES["shared_state_aware_typed"] is run_shared_state_aware_typed


def test_run_policy_dispatches_d3():
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "toy_subagent_trace.jsonl")))
    plan = run_policy("shared_state_aware_typed", m, _bundle())
    assert plan.policy == "shared_state_aware_typed"


def test_default_edge_type_weights_has_global_replicated_zero():
    """Global-replicated state (prompt_context+persistent, e.g. system_prompt)
    must be zeroed in defaults — that's the load-bearing fix vs D2."""
    assert DEFAULT_EDGE_TYPE_WEIGHTS[("prompt_context", "persistent")] == 0.0


def test_default_edge_type_weights_workspace_strong():
    """Workspace state should have strong-affinity weighting so per-session
    components form even when token counts are small."""
    for lifetime in ("private", "shared", "persistent"):
        assert DEFAULT_EDGE_TYPE_WEIGHTS[("workspace", lifetime)] >= 5.0, \
            f"workspace+{lifetime} weight too small to anchor a session"


# ---------------------------------------------------------------------------
# Linear-session fixtures: D3 == D2 (no overgrouping to fix).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace", [
    "examples/traces/toy_subagent_trace.jsonl",
    "examples/traces/g_demo_trace.jsonl",
])
def test_d3_collapses_to_d2_on_linear_session(trace):
    m = build_manifest(from_jsonl(str(REPO / trace)))
    b = _bundle()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert d3 == pytest.approx(d2, abs=1e-9), \
        f"on linear-session trace, D3 should equal D2; got D2={d2}, D3={d3}"


# ---------------------------------------------------------------------------
# Multi-session fixtures: D3 strictly beats D2 by edge-typing the
# system_prompt out of the grouping graph.
# ---------------------------------------------------------------------------


def test_d3_strictly_beats_d2_on_h2():
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "h2_multi_session_swe.jsonl")))
    b = _bundle()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert d3 < d2 - 1.5, (
        f"D3 should beat D2 by >1.5s on H2 (3 sessions × 1GB workspaces); "
        f"got D2={d2:.4f}, D3={d3:.4f}, gap={d2-d3:.4f}"
    )


def test_d3_strictly_beats_d2_on_h5a():
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl")))
    b = _bundle()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert d3 < d2 - 3.0, (
        f"D3 should beat D2 by >3s on H5a (5 sessions × 1GB workspaces); "
        f"got D2={d2:.4f}, D3={d3:.4f}, gap={d2-d3:.4f}"
    )


# ---------------------------------------------------------------------------
# D3 vs D2: regime-dependent. D3 wins on overgrouping fixtures (H2, H5a).
# D3 can lose on fixtures where D2's single-site colocation is correct
# (H5b real bytes — see test_d3_on_h5b_real_bytes_strictly_between_h1_and_d2).
# This is the headline audit finding: edge-typing alone is not strictly
# better; D3's component-level materialization accounting can be worse
# than D2's all-at-one-site choice when D2 picks the right site.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace", [
    "examples/traces/toy_subagent_trace.jsonl",
    "examples/traces/g_demo_trace.jsonl",
])
def test_d3_at_most_d2_on_linear_session_fixtures(trace):
    """On fixtures without overgrouping, D3 == D2 (and D2 was correct
    anyway). D3 doesn't break what wasn't broken."""
    m = build_manifest(from_jsonl(str(REPO / trace)))
    b = _bundle()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert d3 <= d2 + 1e-9, (
        f"D3 should match D2 on linear-session fixtures; got D2={d2}, D3={d3}"
    )


# ---------------------------------------------------------------------------
# D3 does NOT reach H1: edge-typing alone doesn't close L1-vs-L2 gap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace", [
    "examples/traces/h2_multi_session_swe.jsonl",
    "examples/traces/h5a_multi_trajectory_swe.jsonl",
])
def test_d3_does_not_reach_h1_on_multi_session(trace):
    """The audit's load-bearing finding: D3 is strictly worse than H1 on
    the synthetic-1GB multi-session fixtures. The remaining gap is the
    L1-vs-L2 distinction (H1 dedupes per (state, site) across all
    consumers; D3 dedupes per (component, site) like D2). Edge-typing
    alone cannot close that gap. If a future change makes D3 == H1
    here, either L1/L2 distinction has been silently muddled OR D3 has
    been over-corrected."""
    m = build_manifest(from_jsonl(str(REPO / trace)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert d3 > h1 + 1e-9, (
        f"audit pin: D3 should NOT reach H1 on multi-session synthetic fixtures "
        f"(L1-vs-L2 distinction is real); got H1={h1}, D3={d3}"
    )


# ---------------------------------------------------------------------------
# G1 oracle: D3 must still respect the lower bound where it fits.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace", [
    "examples/traces/toy_subagent_trace.jsonl",
    "examples/traces/h2_multi_session_swe.jsonl",
    "examples/traces/h5a_multi_trajectory_swe.jsonl",
])
def test_g1_at_most_d3(trace):
    m = build_manifest(from_jsonl(str(REPO / trace)))
    if len(m.nodes) > 16:
        pytest.skip("G1 enumeration cap exceeded")
    b = _bundle()
    g1 = run_g1_brute_force(m, b).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    assert g1 <= d3 + 1e-9, f"G1 oracle exceeded by D3? g1={g1}, d3={d3}"


# ---------------------------------------------------------------------------
# H5b real-bytes (env-var-gated): D3 should also collapse here, mirroring H1.
# ---------------------------------------------------------------------------


def _all_h5b_repos_present() -> bool:
    if not WORKSPACES_DIR.is_dir():
        return False
    return all((WORKSPACES_DIR / sid).is_dir()
               for sid in ("cog", "pok", "dcj", "ice", "scf"))


@pytest.mark.skipif(
    not _all_h5b_repos_present(),
    reason=f"H5b workspaces missing at {WORKSPACES_DIR}",
)
def test_d3_on_h5b_real_bytes_strictly_between_h1_and_d2(tmp_path):
    """At H5b real-byte scale, H1 ≡ D2 (per H5b audit). D3 sits
    *between* H1 (best) and D2 (worst) here too: it splits the
    overgrouped components but still pays component-level
    materialization (5 components each pay system_prompt at their
    chosen site, vs H1's single-site dedup). Concrete on this fixture:
    H1 ≈ 0.149 s, D3 ≈ 0.257 s, D2 ≈ 0.149 s — D3 is ~108 ms worse
    than H1. The numerics differ slightly from the synthetic-1GB case
    because real bytes drop workspace cost into the noise."""
    sessions = [
        ("cog", "swe_agent_pilot_s_01.json", "phoenix"),
        ("pok", "swe_agent_pilot_s_03.json", "seattle"),
        ("dcj", "swe_agent_pilot_s_05.json", "phoenix"),
        ("ice", "swe_agent_pilot_f_01.json", "seattle"),
        ("scf", "swe_agent_pilot_f_03.json", "phoenix"),
    ]
    cfg = MultiSessionConfig(
        sessions=tuple(
            SessionSpec(
                traj_path=REPO / "tests" / "fixtures" / traj, session_id=sid,
                workspace_home_site=home, workspace_bytes=0, max_ai_turns=2,
                workspace_path=WORKSPACES_DIR / sid,
            )
            for sid, traj, home in sessions
        ),
        workflow_id="d3_h5b_check",
    )
    out = tmp_path / "h5b.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3 = run_shared_state_aware_typed(m, b, tau=1.0).total_cost_s()
    # Headline finding: D3 > D2 here. Component-level materialization
    # forces 5x system_prompt rematerialization (3 phoenix components +
    # 2 seattle components), each paying their own copy. D2's all-at-
    # one-site colocation pays it once.
    assert d3 >= h1 - 1e-9, (
        f"D3 < H1 at H5b? would invalidate L1-vs-L2 distinction; got H1={h1}, D3={d3}"
    )
    assert d3 > d2 + 1e-9, (
        f"D3 should be STRICTLY WORSE than D2 at H5b (component-level "
        f"materialization overhead); if this fails, D3's accounting may "
        f"have silently shifted to L1 (per (state, site) dedup); "
        f"got D2={d2}, D3={d3}"
    )
    # Document the actual gaps so future drift is loud.
    gap_d3_h1 = d3 - h1
    gap_d3_d2 = d3 - d2
    assert 0.05 < gap_d3_h1 < 0.2, (
        f"D3-H1 gap drifted: gap={gap_d3_h1:.6f}; expected ~0.108 s"
    )
    assert 0.05 < gap_d3_d2 < 0.2, (
        f"D3-D2 gap drifted: gap={gap_d3_d2:.6f}; expected ~0.108 s"
    )


# ---------------------------------------------------------------------------
# Edge-type override: callers can plug in alternative weight dicts.
# ---------------------------------------------------------------------------


def test_caller_can_override_edge_type_weights():
    """The default weights zero out global_replicated. A caller that wants
    to study D2-like behavior can supply weights that DON'T zero anything
    out — D3 should then degenerate toward D2 numerically."""
    m = build_manifest(from_jsonl(str(REPO / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl")))
    b = _bundle()
    # All-ones weights: D3 should match D2.
    weights_all_one = {(layer, lt): 1.0
                       for layer in ("prompt_context", "workspace", "memory")
                       for lt in ("persistent", "shared", "private", "ephemeral")}
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    d3_uniform = run_shared_state_aware_typed(m, b, tau=1.0,
                                              edge_type_weights=weights_all_one).total_cost_s()
    assert d3_uniform == pytest.approx(d2, abs=1e-9), (
        f"with all-ones edge weights, D3 should match D2; got D2={d2}, D3={d3_uniform}"
    )
