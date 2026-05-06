"""Tests for the (kv_bytes, link_bps) sensitivity sweep."""
from __future__ import annotations

import csv
from pathlib import Path

from vagrant_agent.cli import sensitivity_main
from vagrant_agent.sensitivity import gap_survival_rate, run_sweep

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "examples" / "traces" / "toy_subagent_trace.jsonl"
MODELS = REPO / "configs" / "model_profiles.yaml"
SITES = REPO / "configs" / "sites_2site.yaml"


def test_run_sweep_produces_one_row_per_grid_point_per_policy(tmp_path: Path):
    kv_bytes_grid = [10_000, 70_656]
    link_bps_grid = [5e9, 25e9, 100e9]
    policies = ["request_level_no_reuse", "request_level_with_site_cache", "shared_state_aware"]
    rows = run_sweep(
        trace_path=TRACE,
        out_dir=tmp_path,
        model_path=MODELS,
        sites_path=SITES,
        model_name="compact_kv",
        kv_bytes_grid=kv_bytes_grid,
        link_bps_grid=link_bps_grid,
        policies=policies,
    )
    expected = len(kv_bytes_grid) * len(link_bps_grid) * len(policies)
    assert len(rows) == expected


def test_sweep_writes_csv_with_load_bearing_columns(tmp_path: Path):
    run_sweep(
        trace_path=TRACE, out_dir=tmp_path, model_path=MODELS, sites_path=SITES,
        model_name="compact_kv",
        kv_bytes_grid=[70_656], link_bps_grid=[5e9],
        policies=["request_level_with_site_cache", "shared_state_aware"],
    )
    csv_path = tmp_path / "sensitivity.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        assert "kv_bytes_per_token" in cols
        assert "link_bps" in cols
        assert "crossover_bps_seattle" in cols
        assert "link_above_crossover" in cols
        assert "policy" in cols
        assert "cost_weighted_duplication_factor" in cols
        assert "gap_robust" in cols


def test_link_above_crossover_flag_matches_b_star(tmp_path: Path):
    """At a grid point where link_bps > 8 * kv_bytes * max_prefill_tok_s, the
    `link_above_crossover` flag must be True; otherwise False. This is the
    bandwidth-crossover identity surfaced as a per-row diagnostic."""
    rows = run_sweep(
        trace_path=TRACE, out_dir=tmp_path, model_path=MODELS, sites_path=SITES,
        model_name="compact_kv",
        kv_bytes_grid=[10_000],          # small KV → low B*
        link_bps_grid=[1e9, 1e12],       # 1 Gbps and 1 Tbps
        policies=["shared_state_aware"],
    )
    by_link = {r["link_bps"]: r["link_above_crossover"] for r in rows}
    assert by_link[1e9] is False  # 1 Gbps below B* = 8*10000*45000 = 3.6 Gbps
    assert by_link[1e12] is True


def test_gap_survival_rate_on_default_toy_at_realistic_link(tmp_path: Path):
    """On the toy trace at realistic single-flow inter-region link (5 Gbps)
    across the bracketing kv_bytes grid, document the gap survival.

    The MVP claim "shared_state_aware beats request_level_with_site_cache" is
    KNOWN to collapse on linear-session traces (per TASKS.md revised gate
    post-H1 — H1 ≡ D2 numerically). So survival rate should be 0% on the toy.
    A non-zero rate would mean the toy is not actually linear-session, OR a
    future fixture broke the documented finding."""
    rows = run_sweep(
        trace_path=TRACE, out_dir=tmp_path, model_path=MODELS, sites_path=SITES,
        model_name="compact_kv",
        kv_bytes_grid=[10_000, 70_656, 327_680],
        link_bps_grid=[5e9, 25e9, 100e9],
        policies=["request_level_with_site_cache", "shared_state_aware"],
        reference_policy="request_level_with_site_cache",
        challenger_policy="shared_state_aware",
    )
    survival = gap_survival_rate(rows)
    # On the toy (linear-session, single component), H1 ≡ D2 to 1e-9.
    # Strictly less is False. Survival rate must be 0% — if a future change
    # makes this nonzero, either (a) the toy got augmented to break linearity
    # (good — update this assertion), or (b) a policy regressed.
    assert survival == 0.0, f"expected 0% survival on toy (H1 ≡ D2 documented), got {survival}"


def test_d1_strictly_loses_to_h1_on_toy(tmp_path: Path):
    """The strawman D1 (request_level_no_reuse) must strictly lose to H1
    (request_level_with_site_cache) on the toy at any link/kv grid point —
    this is the cache-reuse-only finding from TASKS.md."""
    rows = run_sweep(
        trace_path=TRACE, out_dir=tmp_path, model_path=MODELS, sites_path=SITES,
        model_name="compact_kv",
        kv_bytes_grid=[10_000, 70_656, 327_680],
        link_bps_grid=[5e9, 25e9, 100e9],
        policies=["request_level_no_reuse", "request_level_with_site_cache"],
        reference_policy="request_level_no_reuse",
        challenger_policy="request_level_with_site_cache",
    )
    survival = gap_survival_rate(rows)
    assert survival == 1.0, f"H1 must strictly beat D1 at every grid point, got {survival}"


def test_cli_sensitivity_main_smoke(tmp_path: Path):
    rc = sensitivity_main([
        "--trace", str(TRACE),
        "--out", str(tmp_path),
        "--kv-bytes", "70656",
        "--link-bps", "5e9,25e9",
        "--policies", "request_level_with_site_cache,shared_state_aware",
    ])
    assert rc == 0
    assert (tmp_path / "sensitivity.csv").exists()
