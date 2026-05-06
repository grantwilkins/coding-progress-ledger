"""
Claim:
H5a fixture (`examples/traces/h5a_multi_trajectory_swe.jsonl`) is the same
multi_session SWE_style structure as H2 but built from **five distinct real
SWE_agent trajectories** (cached pilot_zero rollouts: cognitive_complexity,
poke_env, dataclasses_json, iceprod, setup_cfg_fmt). H2 reused
`swe_agent_pilot_s_07.json` 3x and so could not distinguish "the H1<D2
mechanism survives real_trajectory variation" from "the mechanism is an
artifact of one_trajectory replay". H5a closes that trajectory_reuse gap.

Workspace bytes remain **synthetic** (1 GB per session, set by the fixture
builder) — H5a does NOT close the synthetic_bytes gap. That is H5b's job
(real bytes via repo clone or rollout_dir capture, deferred).

Numerical anchor: with 2 of 5 workspace homes at the minority site
(seattle: pok, ice; phoenix: cog, dcj, scf), D2 colocates at phoenix and
pays two 1 GB cross_site artifact_copy transfers; H1 places per_session,
keeping each workspace at its home. So D2 - H1 ~= 2 * (8 * 1 GB / 5 Gbps)
= 2 * 1.6 = 3.2 s exactly. Conservative gate: gap > 3.0 s.

Plausible wrong implementations the tests below try to catch:
- fixture builder collapses the 5 trajectories into one (e.g. a stray
  global state map) -> issue_text content_hashes would coincide, masking
  the multi_trajectory framing. Catch via "5 distinct content_hashes."
- a future cleanup pass deletes the per_trajectory fixture files and
  silently breaks regenerability -> catch via explicit existence checks
  on every cached trajectory, before the regenerator runs.
- placement asymmetry breaks (H1 routes a session away from its
  workspace_home) -> catch via a placement_by_session test.
- sensitivity grid corner_flip -> the H5a gap is bytes_layer like H2's,
  scaled 2x; require 100% sign consistency, same as H2.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from agent_migrate_agent import build_manifest
from agent_migrate_agent.adapters.swe_agent_multi import (
    MultiSessionConfig,
    SessionSpec,
    generate_to_file,
)
from agent_migrate_agent.policies import (
    G1_MAX_ENUMERATIONS,
    run_g1_brute_force,
    run_g2_local_search,
    run_request_level_no_reuse,
    run_request_level_with_site_cache,
    run_shared_state_aware,
)
from agent_migrate_agent.profiles import load_bundle

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
CANONICAL_FIXTURE = REPO / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"

CANONICAL_WORKSPACE_BYTES = 1_000_000_000
CANONICAL_LINK_BPS = 5_000_000_000
CANONICAL_AI_TURNS = 2

# (session_id, trajectory file, workspace_home_site). Three at phoenix,
# two at seattle so D2's majority_home colocation pays two cross_site
# artifact_copy transfers (one per minority_home workspace).
CANONICAL_SESSIONS = (
    ("cog", "swe_agent_pilot_s_01.json", "phoenix"),
    ("pok", "swe_agent_pilot_s_03.json", "seattle"),
    ("dcj", "swe_agent_pilot_s_05.json", "phoenix"),
    ("ice", "swe_agent_pilot_f_01.json", "seattle"),
    ("scf", "swe_agent_pilot_f_03.json", "phoenix"),
)


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _canonical_config(workspace_bytes: int = CANONICAL_WORKSPACE_BYTES,
                      max_ai_turns: int = CANONICAL_AI_TURNS,
                      workspace_home_overrides: dict[str, str] | None = None
                      ) -> MultiSessionConfig:
    overrides = workspace_home_overrides or {}
    specs = []
    for sid, traj_name, home in CANONICAL_SESSIONS:
        specs.append(SessionSpec(
            traj_path=FIX / traj_name,
            session_id=sid,
            workspace_home_site=overrides.get(sid, home),
            workspace_bytes=workspace_bytes,
            max_ai_turns=max_ai_turns,
        ))
    return MultiSessionConfig(
        sessions=tuple(specs),
        workflow_id="h5a_multi_trajectory_swe",
        root_task="H5a: 5 distinct SWE-agent trajectories with synthetic workspace bytes",
    )


# ---------------------------------------------------------------------------
# Fixture availability — surface missing trajectory files BEFORE downstream
# tests fail with confusing manifest errors.
# ---------------------------------------------------------------------------


def test_canonical_fixture_committed():
    assert CANONICAL_FIXTURE.exists(), f"missing canonical H5a fixture at {CANONICAL_FIXTURE}"


def test_all_five_trajectory_fixtures_present():
    """Each cached trajectory must be in tests/fixtures/. If a cleanup pass
    deletes one, the regenerator below would surface the failure as a
    `FileNotFoundError`-during_generation, which is recoverable but loud."""
    for _, traj_name, _ in CANONICAL_SESSIONS:
        path = FIX / traj_name
        assert path.exists(), f"missing trajectory fixture: {path}"


# ---------------------------------------------------------------------------
# Multi_trajectory framing — the load_bearing difference vs H2.
# ---------------------------------------------------------------------------


def test_five_issue_text_states_have_distinct_content_hashes():
    """H2's analogue test asserts identical hashes across reused s_07. H5a
    must assert the OPPOSITE: 5 distinct trajectories means 5 distinct
    issue_text content_hashes. If they collapse, the trajectory_reuse gap
    has reopened (e.g. someone copied s_07 into all five fixture slots)."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    issue_ids = [f"issue_text_{sid}" for sid, _, _ in CANONICAL_SESSIONS]
    hashes = {sid: m.state_objects[sid].content_hash for sid in issue_ids}
    assert len(set(hashes.values())) == 5, \
        f"expected 5 distinct issue_text hashes; got {hashes}"


def test_one_shared_system_prompt_with_all_consumers():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    sp = m.state_objects["system_prompt"]
    assert set(sp.consumers) == set(m.nodes)


def test_five_distinct_workspaces_with_expected_homes():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    expected = {f"workspace_{sid}": home for sid, _, home in CANONICAL_SESSIONS}
    actual = {sid: m.state_objects[sid].home_site for sid in expected}
    assert actual == expected
    for sid in expected:
        assert m.state_objects[sid].layer == "workspace"
        assert m.state_objects[sid].bytes == CANONICAL_WORKSPACE_BYTES


def test_workspace_consumers_are_session_local():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    for sid, _, _ in CANONICAL_SESSIONS:
        consumers = m.state_objects[f"workspace_{sid}"].consumers
        for c in consumers:
            assert c.startswith(f"{sid}_"), \
                f"workspace_{sid} leaked to non_session consumer {c!r}"


def test_node_count_matches_sessions_times_ai_turns():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    assert len(m.nodes) == 5 * 2


# ---------------------------------------------------------------------------
# Numerical: H1 < D2 by ~3.2 s (2 minority_home workspaces × 1.6 s each).
# ---------------------------------------------------------------------------


def test_link_bps_matches_canonical_assumption():
    bundle = _bundle()
    link = bundle.link("phoenix", "seattle")
    assert link.effective_bps == CANONICAL_LINK_BPS


def test_h1_strictly_better_than_d2_on_canonical_fixture():
    """Two of five workspaces (pok, ice) live at seattle; D2 colocates at
    phoenix (majority home, 3 sessions) and pays both cross_site
    artifact_copies. H1 places per_session and keeps each workspace local.

        gap >= 2 * 8 * 1 GB / 5 Gbps - small_prompt_terms
            = 2 * 1.6 - O(0.05) ~= 3.15 s
    """
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    gap = d2 - h1
    expected = 2.0 * 8.0 * CANONICAL_WORKSPACE_BYTES / CANONICAL_LINK_BPS
    assert gap > 3.0, f"H5a gap collapsed: gap={gap:.4f}s (expected ~{expected}s)"
    assert gap < expected + 0.2, \
        f"gap={gap:.4f}s exceeds 2*artifact_copy + slack; new term in cost model?"


def test_h1_places_each_session_at_its_workspace_home():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    h1 = run_request_level_with_site_cache(m, _bundle())
    sites = {p.node_id: p.site for p in h1.placements}
    for sid, _, home in CANONICAL_SESSIONS:
        for turn in (1, 2):
            nid = f"{sid}_S{turn}"
            assert sites[nid] == home, f"{nid} expected {home}, got {sites[nid]}"


def test_d2_groups_into_single_component():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    d2 = run_shared_state_aware(m, _bundle(), tau=1)
    assert len(d2.meta["components"]) == 1
    assert set(d2.meta["components"][0]) == set(m.nodes)


def test_d2_places_canonical_component_at_phoenix():
    """3 phoenix workspaces vs 2 seattle. D2 picks the cheaper_total side
    (phoenix), where the saved 3 home_side artifact_copies outweigh the 2
    paid seattle_side ones."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    d2 = run_shared_state_aware(m, _bundle(), tau=1)
    sites = {p.site for p in d2.placements}
    assert sites == {"phoenix"}, f"D2 expected to colocate at phoenix; got {sites}"


def test_d1_strictly_worse_than_h1_on_canonical():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    d1 = run_request_level_no_reuse(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert d1 - h1 > 0.1, f"D1 collapsed to H1: D1={d1:.4f}, H1={h1:.4f}"


def test_h1_and_d1_share_placements():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b)
    d1 = run_request_level_no_reuse(m, b)
    assert {p.node_id: p.site for p in h1.placements} == \
           {p.node_id: p.site for p in d1.placements}


# ---------------------------------------------------------------------------
# G1 oracle: at_least_as_good and fits enumeration cap.
# ---------------------------------------------------------------------------


def test_g1_at_least_as_good_as_h1():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    g1 = run_g1_brute_force(m, b).total_cost_s()
    assert g1 <= h1 + 1e-9


def test_g1_fits_under_enumeration_cap():
    """5 sessions x 2 ai turns = 10 nodes; K=2 sites -> 1024 enumerations
    (well under cap). If the canonical fixture grows past N=16, G1 would
    silently raise; pin the math here."""
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    space = len(b.sites) ** len(m.nodes)
    assert space <= G1_MAX_ENUMERATIONS, \
        f"H5a fixture grew to {len(m.nodes)} nodes; K^N={space} > cap {G1_MAX_ENUMERATIONS}"
    plan = run_g1_brute_force(m, b)
    assert plan.meta["enumerated"] == space


def test_g2_local_search_matches_or_beats_h1():
    m = build_manifest(from_jsonl(str(CANONICAL_FIXTURE)))
    b = _bundle()
    g2 = run_g2_local_search(m, b).total_cost_s()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    assert g2 <= h1 + 1e-9


# ---------------------------------------------------------------------------
# Mechanism (anti_drift): H1 == D2 when no home asymmetry.
# ---------------------------------------------------------------------------


def test_h1_equals_d2_when_no_home_asymmetry(tmp_path: Path):
    cfg = _canonical_config(workspace_home_overrides={
        sid: "phoenix" for sid, _, _ in CANONICAL_SESSIONS
    })
    out = tmp_path / "h5a_no_asymmetry.jsonl"
    generate_to_file(cfg, out)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    assert h1 == pytest.approx(d2, abs=1e-9), \
        f"removing home asymmetry should collapse H1 == D2; got H1={h1}, D2={d2}"


# ---------------------------------------------------------------------------
# Sensitivity: gap survives the bracketing grid (gate_relevant).
# ---------------------------------------------------------------------------


def test_h1_d2_gap_survives_full_sensitivity_grid():
    """Same gate as H2 — bytes_layer gap, scaled 2x; sign consistency
    required across the kv_bytes x link_bps bracketing grid."""
    from agent_migrate_agent.sensitivity import gap_survival_rate, run_sweep
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
    assert gap_survival_rate(rows) >= 0.5
    seen: dict[tuple, float] = {}
    for r in rows:
        key = (r["kv_bytes_per_token"], r["link_bps"])
        if key not in seen:
            seen[key] = r["gap_abs_s"]
    signs = {1 if g > 0 else (-1 if g < 0 else 0) for g in seen.values()}
    assert signs == {1}, f"sign inconsistency across grid; gaps={seen}"


# ---------------------------------------------------------------------------
# Regenerability: fixture must round_trip from canonical config.
# ---------------------------------------------------------------------------


def test_committed_fixture_matches_regeneration(tmp_path: Path):
    fresh = tmp_path / "fresh.jsonl"
    generate_to_file(_canonical_config(), fresh)
    assert fresh.read_bytes() == CANONICAL_FIXTURE.read_bytes(), \
        "committed H5a fixture diverges from canonical config; regenerate or fix the config"


def test_generate_to_file_is_byte_deterministic(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    generate_to_file(_canonical_config(), a)
    generate_to_file(_canonical_config(), b)
    assert a.read_bytes() == b.read_bytes()
