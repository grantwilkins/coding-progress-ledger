"""
Claim:
H2 fixture (`examples/traces/h2_multi_session_swe.jsonl`) is the smallest
realistic SWE-style fixture in which `request_level_with_site_cache` (H1)
strictly beats `shared_state_aware` (D2) on cost-weighted total. The gap is
dominated by exactly one workspace state whose home_site differs from the
component's globally-best site, forcing D2 to pay a 1 GB cross-site
artifact_copy that H1 avoids.

Numerical anchor (derived from configs/sites_2site.yaml + compact_kv):
    artifact_copy(1 GB, 5 Gbps) = 8 * 1e9 / 5e9 = 1.6 s exactly.

Plausible wrong implementations the tests below try to catch:
- adapter dedupes state_ids globally (not per-session) -> issue_text and
  tool_outputs would collapse across sessions, erasing the multi-session
  structure. Catch via "3 distinct issue_text_<sid> with the same hash."
- adapter forgets to set home_site on workspace_<sid> -> D2 collapses to H1
  because there's no asymmetry. Catch via the mechanism test (set all homes
  equal -> equality; canonical mix -> strict inequality).
- adapter declares system_prompt per-session -> would hard-fail at manifest
  build (duplicate state_id) or, if quietly tolerated, the headline gap
  changes magnitude. Catch via "1 system_prompt with all-N consumers."
- policy double-counts cross-site materialization or mis-routes a
  per-session node to the wrong workspace home -> D1/H1 placement test
  catches it.
- G1 silently exceeds enumeration cap -> catch via explicit run-without-
  exception assertion AND meta["enumerated"] check.
- a future site/model config tweak shifts costs but masks the H2 finding ->
  catch via the mechanism test (homes-all-equal collapse), which is a
  property of the FIXTURE STRUCTURE rather than absolute cost values.
- regenerator non-determinism -> catch via byte-identical regen.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.adapters.swe_agent_multi import (
    MultiSessionConfig,
    SessionSpec,
    generate_events,
    generate_to_file,
)
from vagrant_agent.policies import (
    G1_MAX_ENUMERATIONS,
    run_g1_brute_force,
    run_g2_local_search,
    run_request_level_no_reuse,
    run_request_level_with_site_cache,
    run_shared_state_aware,
)
from vagrant_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
TRAJ = REPO / "tests" / "fixtures" / "swe_agent_pilot_s_07.json"
CANONICAL_FIXTURE = REPO / "examples" / "traces" / "h2_multi_session_swe.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"

CANONICAL_WORKSPACE_BYTES = 1_000_000_000
CANONICAL_LINK_BPS = 5_000_000_000  # mirrors sites_2site.yaml; unit-conversion guard below
CANONICAL_AI_TURNS = 2


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _canonical_config(traj_path=TRAJ, workspace_homes=("phoenix", "seattle", "phoenix"),
                     workspace_bytes=CANONICAL_WORKSPACE_BYTES, max_ai_turns=CANONICAL_AI_TURNS):
    return MultiSessionConfig(sessions=tuple(
        SessionSpec(traj_path=traj_path, session_id=sid,
                    workspace_home_site=home, workspace_bytes=workspace_bytes,
                    max_ai_turns=max_ai_turns)
        for sid, home in zip(("sa", "sb", "sc"), workspace_homes)
    ))


def _build_at(tmp_path: Path, **overrides) -> Path:
    """Generate a fresh H2 trace in tmp_path, return the path."""
    cfg = _canonical_config(**overrides)
    out = tmp_path / "h2.jsonl"
    generate_to_file(cfg, out)
    return out


# ---------------------------------------------------------------------------
# Structural invariants — catch adapter bugs before policy bugs.
# ---------------------------------------------------------------------------


def test_canonical_fixture_committed_to_examples():
    """Headline tests below depend on this exact file. If it disappears,
    fail loudly rather than skipping the H2 phenomenon test."""
    assert CANONICAL_FIXTURE.exists(), f"missing canonical H2 fixture at {CANONICAL_FIXTURE}"


def test_one_shared_system_prompt_with_all_consumers():
    """system_prompt MUST be a single state object consumed by every llm_call
    across every session. If the adapter accidentally per-sessions it (e.g.,
    by prefixing the state_id), the multi-session phenomenon evaporates."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    sp = m.state_objects["system_prompt"]
    assert set(sp.consumers) == set(m.nodes), \
        "system_prompt must be consumed by every node; got " \
        f"consumers={sorted(sp.consumers)}, nodes={sorted(m.nodes)}"


def test_three_distinct_workspaces_with_expected_homes():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    expected = {"workspace_sa": "phoenix", "workspace_sb": "seattle", "workspace_sc": "phoenix"}
    actual = {sid: m.state_objects[sid].home_site for sid in expected}
    assert actual == expected
    for sid in expected:
        assert m.state_objects[sid].layer == "workspace"
        assert m.state_objects[sid].bytes == CANONICAL_WORKSPACE_BYTES


def test_workspace_consumers_are_session_local():
    """workspace_<sid> must be consumed only by that session's nodes; if
    consumers leak across sessions, every workspace becomes effectively
    shared and the H1<D2 mechanism is destroyed."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    for sid in ("sa", "sb", "sc"):
        consumers = m.state_objects[f"workspace_{sid}"].consumers
        for c in consumers:
            assert c.startswith(f"{sid}_"), \
                f"workspace_{sid} leaked to non-session consumer {c!r}"


def test_issue_text_states_distinct_with_colliding_hashes():
    """We reuse s_07 across sessions, so issue_text content is byte-identical.
    The adapter MUST keep state_ids distinct so each session has its own
    state object (the 'multi-session' framing); the content_hash collision
    is intentional and proves the per-session prefixing actually triggered."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    issue_ids = ["issue_text_sa", "issue_text_sb", "issue_text_sc"]
    hashes = {sid: m.state_objects[sid].content_hash for sid in issue_ids}
    assert len(set(hashes.values())) == 1, \
        "expected identical content_hash across reused sessions; got distinct hashes — fixture drift"
    assert len({m.state_objects[sid].state_id for sid in issue_ids}) == 3


# ---------------------------------------------------------------------------
# Per-session-scoped tool_output dedup (vs global dedup).
# This is the easiest place for the adapter to drift: a global declared_outputs
# map would silently collapse byte-identical tool replies ACROSS sessions and
# mask the multi-session structure. Construct the smallest fixture that proves
# the dedup is per-session, by feeding a tiny synthetic trajectory whose
# user-replies are byte-identical to s_07's.
# ---------------------------------------------------------------------------


def _two_session_with_repeated_reply_traj(tmp_path: Path) -> Path:
    """Synthetic SWE-agent traj: 3 ai turns, 2 of the user replies share text.
    Used to assert (a) within-session dedup collapses repeats, (b) across-
    session dedup does NOT (state_ids are distinct)."""
    repeated_reply = "the literal same reply text"
    traj = {
        "instance_id": "h2_dedup_probe",
        "model_name": "test",
        "trajectory": [
            {"role": "system", "system_prompt": "SETTING: shared sys", "text": None,
             "cutoff_date": "n/a", "mask": False},
            {"role": "user", "system_prompt": None, "text": "ISSUE: do the thing",
             "cutoff_date": None, "mask": False},
            {"role": "ai", "system_prompt": None, "text": "first ai", "cutoff_date": None, "mask": True},
            {"role": "user", "system_prompt": None, "text": repeated_reply, "cutoff_date": None, "mask": False},
            {"role": "ai", "system_prompt": None, "text": "second ai", "cutoff_date": None, "mask": True},
            {"role": "user", "system_prompt": None, "text": repeated_reply, "cutoff_date": None, "mask": False},
            {"role": "ai", "system_prompt": None, "text": "third ai", "cutoff_date": None, "mask": True},
        ],
    }
    p = tmp_path / "dedup_probe.json"
    p.write_text(json.dumps(traj))
    return p


def test_intra_session_tool_output_dedupes_by_content_hash(tmp_path: Path):
    traj = _two_session_with_repeated_reply_traj(tmp_path)
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=traj, session_id="sa", workspace_home_site="phoenix",
                    workspace_bytes=1_000_000, max_ai_turns=3),
        SessionSpec(traj_path=traj, session_id="sb", workspace_home_site="seattle",
                    workspace_bytes=1_000_000, max_ai_turns=3),
    ))
    out = tmp_path / "probe.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    sa_outputs = [s for s in m.state_objects if s.startswith("tool_output_sa_")]
    assert len(sa_outputs) == 1, \
        f"two byte-identical user replies in one session must collapse to one tool_output; got {sa_outputs}"


def test_cross_session_tool_output_does_not_dedupe(tmp_path: Path):
    """Same content in two sessions must produce two distinct state_ids
    (per-session scoping) — otherwise the multi-session framing fails."""
    traj = _two_session_with_repeated_reply_traj(tmp_path)
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=traj, session_id="sa", workspace_home_site="phoenix",
                    workspace_bytes=1_000_000, max_ai_turns=3),
        SessionSpec(traj_path=traj, session_id="sb", workspace_home_site="seattle",
                    workspace_bytes=1_000_000, max_ai_turns=3),
    ))
    out = tmp_path / "probe.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    sa_set = {s for s in m.state_objects if s.startswith("tool_output_sa_")}
    sb_set = {s for s in m.state_objects if s.startswith("tool_output_sb_")}
    assert sa_set and sb_set
    assert sa_set.isdisjoint(sb_set)
    sa_hashes = {m.state_objects[s].content_hash for s in sa_set}
    sb_hashes = {m.state_objects[s].content_hash for s in sb_set}
    assert sa_hashes == sb_hashes, \
        "cross-session content_hashes should match (proving non-dedup is intentional)"


# ---------------------------------------------------------------------------
# Numerical: H1 < D2 by ~1.6 s, derived from artifact_copy(1 GB, 5 Gbps).
# ---------------------------------------------------------------------------


def test_link_bps_matches_canonical_assumption():
    """The 1.6 s gap below derives from `8 * 1 GB / 5 Gbps`. If anyone
    changes sites_2site.yaml's link bps, the gap derivation breaks; this
    guard catches it before the headline test gives a misleading failure."""
    bundle = _bundle()
    link = bundle.link("phoenix", "seattle")
    assert link.effective_bps == CANONICAL_LINK_BPS, \
        "sites_2site.yaml link bps drifted; update CANONICAL_LINK_BPS and the gap derivation"


def test_h1_strictly_better_than_d2_on_canonical_fixture():
    """Headline H2 finding. Derivation:

        D2 picks ALL nodes at phoenix (single component, link in tau=1 mode).
            workspace_sb (home=seattle) -> cross-site at phoenix
            cost = 8 * 1e9 / 5e9 = 1.6 s.
        H1 places per-session: sa,sc -> phoenix (workspace home), sb -> seattle
            (workspace home). Each workspace at its home costs 0.
        Other state-cost differences (system_prompt and issue_text_sb cross-
        site at seattle) are bounded above by tokens / fast_prefill; for
        compact_kv on sites_2site, they sum to <0.05 s.

        Therefore D2 - H1 >= 1.6 - 0.05 ~= 1.55 s.
    Conservative assertion: gap > 1.5 s.
    """
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    gap = d2 - h1
    expected_workspace_transfer = 8.0 * CANONICAL_WORKSPACE_BYTES / CANONICAL_LINK_BPS
    assert gap > 1.5, f"H1 vs D2 gap collapsed: gap={gap:.4f}s (expected ~{expected_workspace_transfer}s)"
    assert gap < expected_workspace_transfer + 0.1, \
        f"gap={gap:.4f}s exceeds workspace_transfer + slack; new term in cost model?"


def test_h1_places_each_session_at_its_workspace_home():
    """Anti-drift: if H1 (and thus the gap) is to make sense, per-node best-
    site must equal the per-session workspace home_site. If a config tweak
    flips this (e.g. seattle becomes faster enough that even the 1 GB
    transfer is cheaper than staying at phoenix), the H2 mechanism breaks."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    h1 = run_request_level_with_site_cache(m, _bundle())
    sites = {p.node_id: p.site for p in h1.placements}
    for nid in ("sa_S1", "sa_S2", "sc_S1", "sc_S2"):
        assert sites[nid] == "phoenix", f"{nid} expected phoenix, got {sites[nid]}"
    for nid in ("sb_S1", "sb_S2"):
        assert sites[nid] == "seattle", f"{nid} expected seattle, got {sites[nid]}"


def test_d2_groups_into_single_component():
    """Mechanism prerequisite: system_prompt must merge all sessions into a
    single D2 component (otherwise D2 wouldn't be forced to colocate)."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    d2 = run_shared_state_aware(m, _bundle(), tau=1)
    assert len(d2.meta["components"]) == 1
    assert set(d2.meta["components"][0]) == set(m.nodes)


def test_d2_places_canonical_component_at_phoenix():
    """Lock-in: the H1<D2 narrative requires D2 to pick the majority-home
    site (phoenix, with 2/3 sessions anchored there) and pay the seattle
    workspace transfer. If a config tweak silently flips D2 to seattle, the
    gap derivation in test_h1_strictly_better_than_d2_on_canonical_fixture
    would still hold (the OTHER 2 workspaces would transfer instead) but the
    mechanism description in TASKS.md would be wrong."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    d2 = run_shared_state_aware(m, _bundle(), tau=1)
    sites = {p.site for p in d2.placements}
    assert sites == {"phoenix"}, f"D2 expected to colocate at phoenix; got {sites}"


# ---------------------------------------------------------------------------
# Numerical: D1 > H1 (per-consumer materialization vs per-(state, site)).
# ---------------------------------------------------------------------------


def test_d1_strictly_worse_than_h1_on_canonical():
    """D1 pays each (state, consumer) at the consumer's site; H1 pays each
    (state, occupied-site) once. With placement identical, the only
    difference is materialization multiplicity. D1 > H1 by exactly:

        sum over (state, site) of cost_s * (consumer_count_at_site - 1)

    For the canonical fixture, system_prompt alone contributes
        4 phoenix consumers + 2 seattle consumers
        => H1 pays cost_phx + cost_sea
        => D1 pays 4*cost_phx + 2*cost_sea
        => D1 - H1 = 3*cost_phx + 1*cost_sea ~= 3*0.041 + 0.027 ~= 0.15 s

    Plus issue_text and tool_outputs amplify it further. Conservative
    assertion: D1 - H1 > 0.1 s.
    """
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    d1 = run_request_level_no_reuse(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert d1 - h1 > 0.1, f"D1 collapsed to H1: D1={d1:.4f}, H1={h1:.4f}"


def test_h1_and_d1_share_placements():
    """H1 == D1 in placement, differs only in materialization bookkeeping.
    A drift here would mean H1 is no longer 'D1 + per-site cache reuse'."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d1 = run_request_level_no_reuse(m, b)
    assert {p.node_id: p.site for p in h1.placements} == {p.node_id: p.site for p in d1.placements}


# ---------------------------------------------------------------------------
# G1 oracle: at-least-as-good and fits enumeration cap.
# ---------------------------------------------------------------------------


def test_g1_at_least_as_good_as_h1():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    g1 = run_g1_brute_force(m, b).total_cost_s()
    assert g1 <= h1 + 1e-9, f"G1 oracle worse than H1 heuristic: g1={g1}, h1={h1}"


def test_g1_fits_under_enumeration_cap():
    """Truncation (max_ai_turns=2 -> 6 nodes -> 2**6=64 enumerations) is the
    load-bearing reason G1 runs at all on this fixture. If the canonical
    fixture grows and pushes K^N over the cap, the next test (G1 ≤ H1)
    silently turns into 'ValueError raised during run', which is hard to
    diagnose. Pin the cap math explicitly."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    space = len(b.sites) ** len(m.nodes)
    assert space <= G1_MAX_ENUMERATIONS, \
        f"H2 fixture grew to {len(m.nodes)} nodes; K^N={space} exceeds cap {G1_MAX_ENUMERATIONS}"
    plan = run_g1_brute_force(m, b)
    assert plan.meta["enumerated"] == space


def test_g2_local_search_matches_or_beats_h1():
    """G2 seeds from D1; on this fixture it should find the same H1==G1 floor."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    g2 = run_g2_local_search(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert g2 <= h1 + 1e-9


# ---------------------------------------------------------------------------
# Mechanism (anti-drift): the gap is caused BY home_site asymmetry. Setting
# all workspaces to the same home collapses H1 == D2 numerically.
# This is the core "why" assertion. If the gap survives a fixture where the
# only thing changed is workspace homes, then H1<D2 is from something else
# (a bookkeeping bug, or a different mechanism), and the H2 framing is wrong.
# ---------------------------------------------------------------------------


def test_h1_equals_d2_when_no_home_asymmetry(tmp_path: Path):
    out = _build_at(tmp_path, workspace_homes=("phoenix", "phoenix", "phoenix"))
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    assert h1 == pytest.approx(d2, abs=1e-9), \
        f"removing home asymmetry should collapse H1 == D2; got H1={h1}, D2={d2}"


def test_h1_d2_gap_survives_full_sensitivity_grid():
    """Phenomenon-demonstrated gate (TASKS.md ~439-444) requires the gap to
    survive the bracketing grid kv_bytes ∈ {10K, 70656, 327680} ×
    link_bps ∈ {5e9, 25e9, 100e9}. The H2 mechanism is bytes-layer
    (artifact_copy = 8*B/bps), independent of kv_bytes_per_token, so the
    gap shrinks linearly in 1/link_bps but never inverts. We require
    >= 50% gap_robust with sign consistency; in practice we observe 100%."""
    from vagrant_agent.sensitivity import gap_survival_rate, run_sweep
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rows = run_sweep(
            trace_path=CANONICAL_FIXTURE,
            out_dir=td,
            model_path=MODELS,
            sites_path=SITES,
            model_name="compact_kv",
            kv_bytes_grid=[10_000, 70_656, 327_680],
            link_bps_grid=[5e9, 25e9, 100e9],
            policies=["request_level_with_site_cache", "shared_state_aware"],
            reference_policy="shared_state_aware",
            challenger_policy="request_level_with_site_cache",
        )
    survival = gap_survival_rate(rows)
    assert survival >= 0.5, f"H2 gap survives only {survival:.0%} of grid (gate requires >= 50%)"
    seen: dict[tuple, float] = {}
    for r in rows:
        key = (r["kv_bytes_per_token"], r["link_bps"])
        if key not in seen:
            seen[key] = r["gap_abs_s"]
    signs = {1 if g > 0 else (-1 if g < 0 else 0) for g in seen.values()}
    assert signs == {1}, f"sign inconsistency across grid (gate forbids flips); gaps={seen}"


def test_h1_strict_gap_scales_with_workspace_bytes(tmp_path: Path):
    """Direction test: making workspaces larger must (weakly) widen the
    H1 vs D2 gap, because the gap == one cross-site artifact_copy = 8*B/bps.
    Linear relationship; we check monotonicity, not exact slope."""
    b = _bundle()
    gaps = []
    for bytes_ in (10_000_000, 100_000_000, 1_000_000_000):
        out = _build_at(tmp_path / f"ws_{bytes_}", workspace_bytes=bytes_)
        m = build_manifest(from_jsonl(str(out)))
        h1 = run_request_level_with_site_cache(m, b).total_cost_s()
        d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
        gaps.append(d2 - h1)
    assert gaps[0] <= gaps[1] <= gaps[2], f"gap is non-monotone in workspace bytes: {gaps}"
    expected_largest_gap = 8.0 * 1_000_000_000 / CANONICAL_LINK_BPS
    assert gaps[2] == pytest.approx(expected_largest_gap, rel=0.05), \
        f"largest gap {gaps[2]} should match artifact_copy formula {expected_largest_gap}"


# ---------------------------------------------------------------------------
# Adapter contract: identical-system_prompt requirement, regenerability.
# ---------------------------------------------------------------------------


def test_adapter_rejects_mismatched_system_prompts(tmp_path: Path):
    """If two sessions have different system_prompts, the 'shared
    system_prompt' framing fails; the adapter MUST hard-fail rather than
    silently emit two state_declares for the same state_id."""
    other = json.loads(TRAJ.read_text())
    other["trajectory"][0]["system_prompt"] = "DIFFERENT SETTING"
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other))
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=TRAJ, session_id="sa", workspace_home_site="phoenix",
                    workspace_bytes=1_000_000, max_ai_turns=2),
        SessionSpec(traj_path=other_path, session_id="sb", workspace_home_site="seattle",
                    workspace_bytes=1_000_000, max_ai_turns=2),
    ))
    with pytest.raises(ValueError, match="identical system_prompt"):
        generate_events(cfg)


def test_adapter_rejects_zero_or_negative_max_ai_turns():
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=TRAJ, session_id="sa", workspace_home_site="phoenix",
                    workspace_bytes=1, max_ai_turns=0),
        SessionSpec(traj_path=TRAJ, session_id="sb", workspace_home_site="seattle",
                    workspace_bytes=1, max_ai_turns=2),
    ))
    with pytest.raises(ValueError, match="max_ai_turns"):
        generate_events(cfg)


def test_adapter_rejects_duplicate_session_ids():
    cfg = MultiSessionConfig(sessions=(
        SessionSpec(traj_path=TRAJ, session_id="sa", workspace_home_site="phoenix",
                    workspace_bytes=1, max_ai_turns=1),
        SessionSpec(traj_path=TRAJ, session_id="sa", workspace_home_site="seattle",
                    workspace_bytes=1, max_ai_turns=1),
    ))
    with pytest.raises(ValueError, match="unique"):
        generate_events(cfg)


def test_generate_to_file_is_byte_deterministic(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    cfg = _canonical_config()
    generate_to_file(cfg, a)
    generate_to_file(cfg, b)
    assert a.read_bytes() == b.read_bytes()


def test_committed_fixture_matches_regeneration(tmp_path: Path):
    """If someone hand-edits the committed fixture, this test catches it.
    The fixture must be regeneratable byte-for-byte from the canonical config."""
    fresh = tmp_path / "fresh.jsonl"
    generate_to_file(_canonical_config(), fresh)
    assert fresh.read_bytes() == CANONICAL_FIXTURE.read_bytes(), \
        "committed fixture diverges from canonical config; regenerate or fix the config"


# ---------------------------------------------------------------------------
# Truncation: the tool_output orphan-state defense.
# ---------------------------------------------------------------------------


def test_no_orphan_tool_output_states(tmp_path: Path):
    """Truncating to max_ai_turns < trajectory length must not leave dangling
    state_declare events (tool_outputs whose ai-turn consumer was truncated
    away). Orphans are harmless to policies but pollute the manifest."""
    out = _build_at(tmp_path, max_ai_turns=2)
    m = build_manifest(from_jsonl(str(out)))
    for sid, st in m.state_objects.items():
        if sid.startswith("tool_output_"):
            assert len(st.consumers) > 0, \
                f"orphan tool_output state {sid!r} (declared but never read)"


# ---------------------------------------------------------------------------
# Sanity: total node count derives from sessions x max_ai_turns.
# ---------------------------------------------------------------------------


def test_node_count_matches_sessions_times_ai_turns(tmp_path: Path):
    out = _build_at(tmp_path, max_ai_turns=2)
    m = build_manifest(from_jsonl(str(out)))
    assert len(m.nodes) == 3 * 2

    by_session = Counter(nid.split("_")[0] for nid in m.nodes)
    assert by_session == Counter({"sa": 2, "sb": 2, "sc": 2})
