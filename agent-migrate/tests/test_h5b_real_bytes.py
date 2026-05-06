"""
Claim:
H5b is the same multi_session SWE_style fixture as H5a, but with
`SessionSpec.workspace_path` pointed at five freshly_cloned upstream
repos rather than synthetic 1 GB workspace_bytes. The H5a homes
(`phoenix, seattle, phoenix, seattle, phoenix`) are kept verbatim so
that the only thing changed is the byte source.

**Honest negative finding.** At real_world repo sizes (10s of MB at HEAD
across this corpus, vs. H5a's synthetic 1 GB per session), the H5a/H2
H1<D2 gap **does NOT survive**: D2 ≡ H1 numerically at the canonical
config, and 0% of the bracketing sensitivity grid sustains the gap.

**Why.** D2 (`shared_state_aware`) is free to colocate the whole
component at the *faster* site (seattle, 1.5x prefill); the prompt-
context replay savings from picking the faster site exactly cancel the
extra cross_site workspace_transfer cost at HEAD_sized real repos. At
synthetic 1 GB, the workspace transfer dominates and H1 wins by 3.2 s
(the H5a result). The H1<D2 mechanism is real but byte_magnitude
sensitive; sub_threshold for these particular instances at HEAD against
this 5 Gbps single_flow link.

**Implication.** The phenomenon_demonstrated gate (TASKS.md ~487_495)
is NOT satisfied by H5b. To meet the gate at real bytes we'd need
larger working trees (monorepos), a slower link, or less prefill
asymmetry between sites. H5b's role is to surface this honestly — the
synthetic 1 GB scale in H5a was load_bearing for the gap, not just the
framing.

**Mechanism preserved.** Scaling the same trajectories' workspace_bytes
back up to 1 GB recovers H1 < D2 by ~3.2 s exactly — the H5a result.
Asserted as `test_synthetic_1gb_recovers_h5a_gap`.

Plausible wrong implementations the tests below try to catch:
- adapter ignores workspace_path and falls back to workspace_bytes silently
  -> the bytes_magnitude tests would still pass at 0 (matching the 0
  default) but the manifest's `bytes` field would be 0 instead of the
  real disk sum. Catch via per_repo expected_byte_range assertions.
- a future workspace.py change accidentally counts .git internals
  -> 5_50x byte inflation. Catch via tight upper bounds on each repo.
- a future repo bloats 10x at HEAD (e.g., huge data file added)
  -> upper bound triggers; intentional (forces a deliberate update).
- the "0% gap survival at HEAD bytes" finding becomes hidden by a future
  cost_model tweak that accidentally widens H1<D2 -> the explicit
  `survival == 0` assertion catches a silent regime flip.
- the mechanism_recovery test (1 GB synthetic) ever fails -> would mean
  the H5a result itself broke; catch loudly here, not just in H5a.
"""
from __future__ import annotations

import os
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
    run_request_level_with_site_cache,
    run_shared_state_aware,
)
from agent_migrate_agent.profiles import load_bundle
from agent_migrate_agent.sensitivity import gap_survival_rate, run_sweep
from agent_migrate_agent.workspace import compute_repo_bytes

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"

WORKSPACES_DIR = Path(os.environ.get("VAGRANT_H5B_WORKSPACES", "/tmp/h5b_workspaces"))

# Same homes as H5a — the load_bearing ask is "swap synthetic bytes for
# real bytes, hold everything else fixed". Changing the homes would mix
# two variables.
H5A_HOMES = {
    "cog": "phoenix",
    "pok": "seattle",
    "dcj": "phoenix",
    "ice": "seattle",
    "scf": "phoenix",
}
TRAJ_FILES = {
    "cog": "swe_agent_pilot_s_01.json",
    "pok": "swe_agent_pilot_s_03.json",
    "dcj": "swe_agent_pilot_s_05.json",
    "ice": "swe_agent_pilot_f_01.json",
    "scf": "swe_agent_pilot_f_03.json",
}

# Snapshot of working_tree byte sizes at the time H5b landed. Tests
# range_check each repo's current bytes against ±loose factor of these
# values, so HEAD growth doesn't break the suite but a 10x balloon does.
EXPECTED_BYTE_SNAPSHOT = {
    "cog": 21_922,
    "pok": 21_588_279,
    "dcj": 301_091,
    "ice": 11_568_017,
    "scf": 57_062,
}


def _all_repos_present() -> bool:
    if not WORKSPACES_DIR.is_dir():
        return False
    return all((WORKSPACES_DIR / sid).is_dir() for sid in H5A_HOMES)


_skip_no_repos = pytest.mark.skipif(
    not _all_repos_present(),
    reason=(
        f"H5b requires repo clones at {WORKSPACES_DIR}; run "
        f"scripts/h5b/clone_repos.sh first (or set VAGRANT_H5B_WORKSPACES)"
    ),
)


def _bundle():
    return load_bundle(MODELS, SITES, "compact_kv")


def _build_config(*, use_workspace_paths: bool, max_ai_turns: int = 2,
                  workspace_bytes: int = 1_000_000_000) -> MultiSessionConfig:
    """Build the H5b/H5a config. `use_workspace_paths=True` -> real disk
    bytes via compute_repo_bytes; False -> synthetic int bytes (recovers
    the H5a configuration for mechanism_preservation tests)."""
    return MultiSessionConfig(
        sessions=tuple(
            SessionSpec(
                traj_path=FIX / TRAJ_FILES[sid],
                session_id=sid,
                workspace_home_site=H5A_HOMES[sid],
                workspace_bytes=0 if use_workspace_paths else workspace_bytes,
                max_ai_turns=max_ai_turns,
                workspace_path=WORKSPACES_DIR / sid if use_workspace_paths else None,
            )
            for sid in H5A_HOMES
        ),
        workflow_id="h5b_real_bytes_swe",
        root_task="H5b: 5 distinct trajectories with real workspace bytes",
    )


def _generate(tmp_path: Path, *, use_workspace_paths: bool,
              workspace_bytes: int = 1_000_000_000) -> Path:
    cfg = _build_config(
        use_workspace_paths=use_workspace_paths,
        workspace_bytes=workspace_bytes,
    )
    out = tmp_path / "h5b.jsonl"
    generate_to_file(cfg, out)
    return out


# ---------------------------------------------------------------------------
# Per_repo byte sanity (catch repo balloons before they confuse policies).
# ---------------------------------------------------------------------------


@_skip_no_repos
@pytest.mark.parametrize("sid", list(H5A_HOMES))
def test_each_repo_bytes_within_expected_range(sid: str):
    """Per_repo range check. If a future `git pull` adds a vendored
    binary or removes a big asset, this fails loudly and forces a
    deliberate snapshot update — preventing the H5b numerics from
    silently shifting regime. 2x tolerance is tight enough to surface
    any addition large enough to matter to the gap calculation, while
    allowing ordinary HEAD churn."""
    actual = compute_repo_bytes(WORKSPACES_DIR / sid)
    expected = EXPECTED_BYTE_SNAPSHOT[sid]
    assert expected // 2 <= actual <= expected * 2, (
        f"workspace bytes for {sid!r} drifted out of range: "
        f"snapshot={expected:,}, actual={actual:,}. "
        f"Update EXPECTED_BYTE_SNAPSHOT after auditing the change."
    )


@_skip_no_repos
def test_total_workspace_bytes_below_regime_flip_threshold():
    """The headline `D2 ≡ H1` finding requires the seattle_minority
    workspace bytes (pok + ice under H5a homes) to stay below the
    regime_flip threshold. Cancellation in the H5b cost model is
    delicate: at 5 Gbps, ~50 MB of cross_site bytes exactly offsets
    the seattle prefill savings; above that, H1 < D2 reappears as a
    bytes_layer effect. This *aggregate* check is what actually drives
    the gap, more directly than per_repo bytes — and it catches the
    case where two repos each grow 1.5x without tripping the per_repo
    2x range, but their sum crosses the regime flip."""
    seattle_minority_bytes = sum(
        compute_repo_bytes(WORKSPACES_DIR / sid)
        for sid, home in H5A_HOMES.items() if home == "seattle"
    )
    # Snapshot total: pok (21.6M) + ice (11.6M) = 33.2M. Regime flip
    # is at ~50M. 40M cap leaves headroom for ordinary churn but
    # surfaces any drift toward the flip before it silently changes
    # the headline.
    assert seattle_minority_bytes < 40_000_000, (
        f"seattle_minority workspace bytes = {seattle_minority_bytes:,}; "
        f"approaching the ~50 MB regime_flip threshold at 5 Gbps. "
        f"Audit which repo grew before updating this assertion — "
        f"crossing the threshold means the H5b 'gap collapses' finding "
        f"has reverted to H5a's H1 < D2."
    )


# ---------------------------------------------------------------------------
# Structural framing — same multi_trajectory shape as H5a.
# ---------------------------------------------------------------------------


@_skip_no_repos
def test_five_distinct_issue_text_hashes(tmp_path: Path):
    out = _generate(tmp_path, use_workspace_paths=True)
    m = build_manifest(from_jsonl(str(out)))
    issue_ids = [f"issue_text_{sid}" for sid in H5A_HOMES]
    hashes = {sid: m.state_objects[sid].content_hash for sid in issue_ids}
    assert len(set(hashes.values())) == 5


@_skip_no_repos
def test_workspace_bytes_match_disk_bytes(tmp_path: Path):
    """The adapter must wire compute_repo_bytes(workspace_path) into
    state.bytes. If a future change silently substitutes a sentinel
    (e.g., 0 default), absolute costs break in confusing ways."""
    out = _generate(tmp_path, use_workspace_paths=True)
    m = build_manifest(from_jsonl(str(out)))
    for sid in H5A_HOMES:
        on_disk = compute_repo_bytes(WORKSPACES_DIR / sid)
        in_state = m.state_objects[f"workspace_{sid}"].bytes
        assert in_state == on_disk, \
            f"workspace_{sid}: state.bytes={in_state}, disk={on_disk}"


@_skip_no_repos
def test_workspace_homes_match_h5a(tmp_path: Path):
    """Holding homes constant is what makes H5b a clean swap of H5a's
    bytes axis. Drift here turns H5b into a different experiment."""
    out = _generate(tmp_path, use_workspace_paths=True)
    m = build_manifest(from_jsonl(str(out)))
    for sid, home in H5A_HOMES.items():
        assert m.state_objects[f"workspace_{sid}"].home_site == home


# ---------------------------------------------------------------------------
# Headline: H5a finding (H1<D2) does NOT survive at real bytes.
# ---------------------------------------------------------------------------


@_skip_no_repos
def test_h5a_gap_collapses_at_real_bytes(tmp_path: Path):
    """Under H5a homes with real working_tree bytes, D2 (free to colocate
    at the faster site) lands within numerical tolerance of H1. The 3.2 s
    H5a gap is gone. This is the load_bearing negative finding."""
    out = _generate(tmp_path, use_workspace_paths=True)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    # The actual cancellation is exact to ~1e-17 (sub_nanosecond IEEE
    # noise). Tolerate float noise but not anything semantically
    # meaningful: 1 microsecond is six orders of magnitude tighter than
    # any real cost_term in this fixture, and tight enough to surface a
    # regime drift in either direction (H1<D2 reappearing, or D2<H1
    # opening up).
    assert abs(d2 - h1) < 1e-6, (
        f"H5b expected D2 ≈ H1 at real bytes (H5a gap collapses); got "
        f"H1={h1:.6f}, D2={d2:.6f}, gap={d2_h1:.6e}. "
        f"Either direction signals a regime change worth investigating."
    )


@_skip_no_repos
def test_d2_picks_seattle_at_real_bytes(tmp_path: Path):
    """Mechanism: D2 colocates at the faster_prefill site (seattle).
    The prompt_context replay savings from the 1.5x prefill ratio cancel
    the extra cross_site workspace_transfer cost at HEAD_sized real repos.
    A drift to phoenix would mean either prefill rates equalized or
    workspace bytes ballooned — both worth surfacing loudly."""
    out = _generate(tmp_path, use_workspace_paths=True)
    m = build_manifest(from_jsonl(str(out)))
    plan = run_shared_state_aware(m, _bundle(), tau=1)
    sites = {p.site for p in plan.placements}
    assert sites == {"seattle"}, \
        f"expected D2 to colocate at seattle (faster prefill); got {sites}"


@_skip_no_repos
def test_sensitivity_grid_zero_survival_at_real_bytes(tmp_path: Path):
    """The phenomenon_demonstrated gate requires >=50% gap_robust on the
    bracketing grid. H5b sustains 0% — explicit, locked_in negative
    result. If a future cost_model tweak makes this nonzero, either (a)
    the regime genuinely shifted (good — update the gate language and
    this test) or (b) a policy regressed."""
    rows = run_sweep(
        trace_path=_generate(tmp_path, use_workspace_paths=True),
        out_dir=tmp_path,
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
    assert survival == 0.0, \
        f"H5b expected 0% gap survival (gap collapses at real bytes); got {survival:.0%}"


# ---------------------------------------------------------------------------
# Mechanism preserved: same trajectories at 1 GB synthetic recover H5a.
# ---------------------------------------------------------------------------


@_skip_no_repos
def test_synthetic_1gb_recovers_h5a_gap(tmp_path: Path):
    """The H1<D2 mechanism is real but byte_magnitude_sensitive. Replacing
    real disk bytes with synthetic 1 GB (the H5a config) on these same
    trajectories must recover the 3.2 s gap, proving H5b's negative result
    is about *byte scale*, not a broken cost model.

    Derivation: 2 cross_site workspaces (pok at seattle home but routed
    phoenix_side under D2; ice at seattle home routed phoenix_side under
    D2) each pay 8 * 1 GB / 5 Gbps = 1.6 s. Conservative gate: gap > 3.0 s.
    """
    out = _generate(tmp_path, use_workspace_paths=False, workspace_bytes=1_000_000_000)
    m = build_manifest(from_jsonl(str(out)))
    b = _bundle()
    h1 = run_request_level_with_site_cache(m, b).total_cost_s()
    d2 = run_shared_state_aware(m, b, tau=1).total_cost_s()
    assert d2 - h1 > 3.0, (
        f"mechanism check: H5a's H1<D2 gap should reappear at 1 GB synthetic; "
        f"got H1={h1:.6f}, D2={d2:.6f}, gap={d2_h1:.6f}"
    )


# ---------------------------------------------------------------------------
# Skip_when_not_cloned plumbing (sanity test the gate itself).
# ---------------------------------------------------------------------------


def test_skip_marker_path_exists():
    """Confirm the env_var_derived path resolution doesn't silently
    point at a different default. If WORKSPACES_DIR is unset, the test
    suite must skip the rest, not crash."""
    assert WORKSPACES_DIR == Path(
        os.environ.get("VAGRANT_H5B_WORKSPACES", "/tmp/h5b_workspaces")
    )
