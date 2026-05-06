from __future__ import annotations

"""
Claim:
R3 runs the K8 sweep once per ModelProfile under a *model-aware budget*
(prefill capacity scales with the architecture's per-stream prefill rate;
link bandwidth is held constant), and the per-cell flip table distinguishes
architecture-driven regime shifts from no-op axes.

Plausible wrong implementations:
- the per-model loop reuses one bundle so kv_bytes_per_token never changes;
- the budget is held constant across models, so prefill rate variation
  (8x across architectures per FINDINGS) is invisible to K4;
- the flip detector compares stale rows or skips cells when one model is
  missing data (which would silently mark every cell as non-flipping);
- the artifact omits per-model columns so a reviewer cannot tell which
  architecture chose which policy;
- profile YAML changes silently break model-load (no profile-specific kv_bpt
  or prefill rate surfaces in the bundle).
"""

from pathlib import Path

import pytest

from vagrant_agent.k8_regime import RegimeCell, default_bundle
from vagrant_agent.r3_model_sweep import (
    R3_DEFAULT_PROFILES,
    make_r3_budget,
    run_r3_cell,
    run_r3_sweep,
    summarize_r3,
    write_r3_artifacts,
)


REPO = Path(__file__).resolve().parent.parent


def test_default_profiles_load_with_distinct_kv_sizes_and_prefill_rates():
    """Claim: the R3 default profile set is real architectural spread on
    BOTH KV size and prefill rate, not five renames of the same numbers."""
    bundles = [default_bundle(REPO, name) for name in R3_DEFAULT_PROFILES]
    kvs = [b.model.kv_bytes_per_token for b in bundles]
    rates = [b.model.single_stream_prefill_tok_s for b in bundles]
    # Every profile carries a distinct kv_bpt: these must come from distinct
    # architectures, not aliases.
    assert len(set(kvs)) == len(kvs), kvs
    # Prefill rates must all be distinct after the prefill-breakeven update —
    # each architecture has its own attention-FLOPS profile.
    assert len(set(rates)) == len(rates), rates
    # kv_bpt spread must span ≥30× (frontier_v4 ~10K to vanilla_gqa ~328K).
    assert max(kvs) / min(kvs) > 30.0, kvs
    # Prefill rate spread must span ≥30× (frontier_v4 ~5.6k to qwen3_next ~175k).
    assert max(rates) / min(rates) > 30.0, rates


def test_make_r3_budget_scales_prefill_with_model_rate():
    """Claim: prefill capacity is rescaled by the model's per-stream rate
    relative to the K8 baseline (compact_kv). A model with 10x faster
    per-stream prefill must see 10x the per-site token rate at the same
    `prefill_capacity` axis value."""
    cell = RegimeCell(
        n_workflows=10,
        state_scale="tiny",
        prefill_capacity="moderate",
        link_gbps=25,
    )
    baseline = default_bundle(REPO, "compact_kv")
    qwen = default_bundle(REPO, "qwen3_next_hybrid")
    baseline_rate = baseline.model.single_stream_prefill_tok_s
    base_budget = make_r3_budget(cell, baseline, baseline_prefill_tok_s=baseline_rate)
    qwen_budget = make_r3_budget(cell, qwen, baseline_prefill_tok_s=baseline_rate)
    expected_ratio = (
        qwen.model.single_stream_prefill_tok_s
        / baseline.model.single_stream_prefill_tok_s
    )
    base_seattle = base_budget.prefill_tok_s_per_site["seattle"]
    qwen_seattle = qwen_budget.prefill_tok_s_per_site["seattle"]
    assert qwen_seattle == pytest.approx(base_seattle * expected_ratio, rel=1e-9)
    # Link bandwidth is infrastructure, not architecture, and must not move.
    assert (
        base_budget.network_bps_per_link[("phoenix", "seattle")]
        == qwen_budget.network_bps_per_link[("phoenix", "seattle")]
    )


def test_compact_kv_is_the_k8_baseline_unchanged_budget():
    """Claim: under the K8 baseline (compact_kv), the R3 budget collapses to
    the K8 budget — so existing K8 tests and artifacts remain comparable."""
    from vagrant_agent.k8_regime import make_k8_budget

    cell = RegimeCell(
        n_workflows=10, state_scale="tiny", prefill_capacity="tight", link_gbps=5,
    )
    bundle = default_bundle(REPO, "compact_kv")
    r3 = make_r3_budget(cell, bundle)
    k8 = make_k8_budget(cell)
    assert r3.prefill_tok_s_per_site == k8.prefill_tok_s_per_site
    assert r3.network_bps_per_link == k8.network_bps_per_link


def test_r3_sweep_returns_per_model_metrics_for_two_profiles():
    """Claim: model-name keys map to actually different sweep results — i.e.,
    the per-model loop is not a no-op alias."""
    metrics = run_r3_sweep(
        REPO,
        model_names=("compact_kv", "glm_5_mla"),
        n_values=(10,),
        state_scales=("tiny",),
        prefill_caps=("tight",),
        link_gbps_values=(1, 100),
    )
    assert set(metrics) == {"compact_kv", "glm_5_mla"}
    for rows in metrics.values():
        assert len(rows) == 12  # 6 policies × 2 link cells
    by_key_compact = {(r.cell.cell_id, r.policy): r.p50_resume_s for r in metrics["compact_kv"]}
    by_key_glm = {(r.cell.cell_id, r.policy): r.p50_resume_s for r in metrics["glm_5_mla"]}
    assert by_key_compact.keys() == by_key_glm.keys()
    assert any(
        by_key_compact[k] != by_key_glm[k] for k in by_key_compact
    ), "GLM-4.6 GQA should diverge from compact MLA on at least one (cell, policy)"


def test_summarize_r3_marks_flips_when_best_policy_disagrees():
    """Claim: summarize_r3 detects per-cell disagreement across models."""
    metrics = run_r3_sweep(
        REPO,
        model_names=("compact_kv", "glm_5_mla"),
        n_values=(10,),
        state_scales=("tiny",),
        prefill_caps=("tight", "loose"),
        link_gbps_values=(1, 100),
    )
    rows = summarize_r3(metrics)
    assert len(rows) == 4
    for row in rows:
        assert set(row.best_policy_by_model) == {"compact_kv", "glm_5_mla"}
        differs = (
            row.best_policy_by_model["compact_kv"]
            != row.best_policy_by_model["glm_5_mla"]
        )
        assert row.best_policy_flips is differs


def test_write_r3_artifacts_emits_per_model_columns(tmp_path):
    """Claim: the CSV exposes both flip flags and per-model columns so a
    reader can attribute any flip to a specific architecture."""
    metrics = run_r3_sweep(
        REPO,
        model_names=("compact_kv", "qwen3_next_hybrid"),
        n_values=(10,),
        state_scales=("tiny",),
        prefill_caps=("moderate",),
        link_gbps_values=(25,),
    )
    write_r3_artifacts(metrics, tmp_path)
    text = (tmp_path / "r3_regime_by_model.csv").read_text()
    assert "best_policy_compact_kv" in text
    assert "best_policy_qwen3_next_hybrid" in text
    assert "bottleneck_compact_kv" in text
    assert "best_policy_flips" in text
    assert "bottleneck_flips" in text
    assert (tmp_path / "r3_flip_summary.json").exists()
    assert (tmp_path / "r3_flip_counts_by_axis.json").exists()


def test_kv_byte_size_visible_in_per_action_costs():
    """Claim: changing the model profile actually changes the cost K4 sees
    for KV transfer.

    `kv_all` ships every prompt-context state via KV. The K4 result for
    GLM-4.6-class GQA (376,832 bytes/tok) must exceed compact MLA
    (70,656 bytes/tok) on the same episode + link by a factor that tracks
    the kv_bpt ratio. We pin to within 30% of the ratio: the tolerance
    accounts for shared replay/workspace cost on warm-hit states and the
    site-prefill rescaling that affects fallback CONTEXT_REPLAY paths.
    """
    cell = RegimeCell(
        n_workflows=10,
        state_scale="tiny",
        prefill_capacity="loose",
        link_gbps=5,
    )
    compact_bundle = default_bundle(REPO, "compact_kv")
    glm_bundle = default_bundle(REPO, "glm_5_mla")
    baseline_rate = compact_bundle.model.single_stream_prefill_tok_s
    compact_metrics = run_r3_cell(cell, compact_bundle, baseline_prefill_tok_s=baseline_rate)
    glm_metrics = run_r3_cell(cell, glm_bundle, baseline_prefill_tok_s=baseline_rate)
    compact_kv_all = next(r for r in compact_metrics if r.policy == "kv_all")
    glm_kv_all = next(r for r in glm_metrics if r.policy == "kv_all")
    expected_ratio = (
        glm_bundle.model.kv_bytes_per_token
        / compact_bundle.model.kv_bytes_per_token
    )
    actual_ratio = glm_kv_all.p50_resume_s / compact_kv_all.p50_resume_s
    # Direction must hold strictly: heavier KV → slower KV transfer.
    assert glm_kv_all.p50_resume_s > compact_kv_all.p50_resume_s
    # Magnitude must track within 30% of the kv_bpt ratio.
    assert 0.7 * expected_ratio <= actual_ratio <= 1.3 * expected_ratio, (
        f"actual={actual_ratio:.2f}, expected≈{expected_ratio:.2f}"
    )
