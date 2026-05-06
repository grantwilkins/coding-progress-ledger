from __future__ import annotations

"""
Claim:
`run_w_r3_matrix` cross-runs every (anchor, model_profile, cell) and
classifies each into one of the four regimes. Where the observed regime
differs from the compact_kv baseline, the anchor's hypothesis is no longer
profile-independent.

Plausible wrong implementations:
- The matrix runner reuses one bundle so model architecture has no effect
  (same regime for every model on every cell).
- The cell→budget mapping is hard-coded for compact_kv (e.g., uses
  `from_bundle(bundle)` and ignores the cell's prefill_capacity axis).
- The flip detector compares the wrong baseline (e.g., the first row in
  alphabetic profile order instead of compact_kv).
- A profile that should produce a workspace-bound regime (light-KV
  hybrid) is silently labeled prefill (= bottleneck propagation bug).
"""

from pathlib import Path

import pytest

from vagrant_agent.w_under_r3 import (
    AnchorR3Cell,
    W_R3_CELLS,
    run_w_r3_matrix,
    write_w_r3_artifacts,
)
from vagrant_agent.workloads import (
    W1_LARGE_REPO_CODING,
    W2_DATA_RAG_HEAVY,
    W3_MULTI_AGENT_FANOUT,
    REGIMES,
)


REPO = Path(__file__).resolve().parent.parent


def test_w_r3_matrix_covers_full_cross_product():
    """Claim (admissibility): the matrix runner emits exactly one row per
    (anchor × model × cell). A bug that skips a profile or a cell would
    silently miss flip detection."""
    rows = run_w_r3_matrix(
        REPO,
        anchors=(W1_LARGE_REPO_CODING,),
        model_names=("compact_kv", "frontier_v4_fp8"),
        cells=(AnchorR3Cell("medium", "loose", 1),),
        n_workflows=4,
    )
    assert len(rows) == 1 * 2 * 1
    assert {r.model_profile for r in rows} == {"compact_kv", "frontier_v4_fp8"}
    assert {r.anchor_name for r in rows} == {"w1_large_repo_coding"}


def test_w_r3_classification_uses_per_profile_bundle():
    """Claim (correctness): different model profiles must produce
    different classifications when the cell stresses an architecture-
    sensitive resource. Catches a bug where the runner uses one bundle
    for all rows.

    We pick a cell whose bottleneck label is known to flip across profiles
    in the R3 pilot: tiny-state cells where light-KV hybrids make
    network/prefill-cheap, leaving workspace as bottleneck. Use W2 (data
    RAG, large must_move bytes) to exercise this — under different
    architectures the dominant bottleneck and gap shift."""
    rows = run_w_r3_matrix(
        REPO,
        anchors=(W2_DATA_RAG_HEAVY,),
        model_names=("vanilla_gqa_fp16", "qwen3_next_hybrid"),
        cells=(AnchorR3Cell("medium", "tight", 5),),
        n_workflows=4,
    )
    by_model = {r.model_profile: r for r in rows}
    # P50 must differ when KV cost differs by 13× (327680 vs 24576) and
    # prefill rate differs by 17× (10221 vs 175316).
    p50_gqa = by_model["vanilla_gqa_fp16"].classification.strong_reuse_p50_resume_s
    p50_qwen = by_model["qwen3_next_hybrid"].classification.strong_reuse_p50_resume_s
    assert p50_gqa != pytest.approx(p50_qwen, rel=1e-6), (
        f"{p50_gqa=} == {p50_qwen=} — runner is not propagating model profile"
    )


def test_w_r3_observed_regime_is_in_known_set():
    """Claim (taxonomy invariant): every classification must be one of
    the four declared regimes. Catches a string-typo bug (e.g.,
    'state-locality') that would slip through if no schema check."""
    rows = run_w_r3_matrix(
        REPO,
        anchors=(W1_LARGE_REPO_CODING, W3_MULTI_AGENT_FANOUT),
        model_names=("compact_kv", "frontier_v4_fp8"),
        cells=(AnchorR3Cell("medium", "loose", 1),),
        n_workflows=4,
    )
    for row in rows:
        assert row.classification.observed_regime in REGIMES, row


def test_write_w_r3_artifacts_emits_per_profile_rows(tmp_path):
    """Claim (artifact integrity): the per-row CSV has every
    (anchor, profile, cell) combination so a reviewer can audit the
    flip table. The summary JSON reports flips relative to the
    compact_kv baseline."""
    rows = run_w_r3_matrix(
        REPO,
        anchors=(W1_LARGE_REPO_CODING,),
        model_names=("compact_kv", "qwen3_next_hybrid"),
        cells=(AnchorR3Cell("medium", "loose", 100),),
        n_workflows=4,
    )
    write_w_r3_artifacts(rows, tmp_path)
    csv_text = (tmp_path / "w_under_r3.csv").read_text()
    assert "compact_kv" in csv_text
    assert "qwen3_next_hybrid" in csv_text
    assert "matches_hypothesis" in csv_text
    summary_path = tmp_path / "w_under_r3_summary.json"
    assert summary_path.exists()
    import json
    summary = json.loads(summary_path.read_text())
    assert "regime_flips_vs_compact_kv_baseline" in summary
    assert summary["model_profiles"][0] == "compact_kv"


def test_w_r3_default_cells_cover_three_capacity_settings():
    """Claim (boundary): the default cell tuple spans a slow-link cell, a
    multi-resource cell, and a fast-link cell so an anchor whose regime
    only flips in one of those is still detected."""
    assert len(W_R3_CELLS) >= 3
    link_speeds = {c.link_gbps for c in W_R3_CELLS}
    assert len(link_speeds) >= 2, link_speeds  # at least two distinct links
    prefill_modes = {c.prefill_capacity for c in W_R3_CELLS}
    assert len(prefill_modes) >= 2, prefill_modes
