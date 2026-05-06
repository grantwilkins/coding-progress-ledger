from __future__ import annotations

"""
Claim:
K8's aggregate estimator is a regime_map approximation over the same cells,
policies, and resource budgets as exact K4, and calibration artifacts compare
those quantities without mixing episodes or axes.

Plausible wrong implementations:
- exact and aggregate runs generate different episodes for the same cell;
- calibration compares only shapes and misses policy_level p50/bottleneck drift;
- link bandwidth or prefill capacity axes are ignored in the ResourceBudget;
- aggregate evaluation double_counts shared materialization or warm reuse;
- artifacts omit the cell metadata needed to interpret calibration rows.
"""

from pathlib import Path

from agent_migrate_agent.k8_regime import (
    RegimeCell,
    calibrate_k8_estimator,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
    run_k8_cell,
    summarize_cells,
    write_k8_calibration_artifacts,
)


REPO = Path(__file__).resolve().parent.parent


def test_k8_cell_runs_fixed_policy_set_and_identifies_best_policy():
    """Claim: K8 produces a per_cell regime comparison over the fixed
    policy set, not just another one_off mixed_vs_baseline fixture."""
    bundle = default_bundle(REPO)
    cell = RegimeCell(
        n_workflows=10,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=5,
        seed=8100,
    )
    rows = run_k8_cell(cell, bundle)
    assert {row.policy for row in rows} == {
        "strong_reuse",
        "replay_all",
        "kv_all",
        "workspace_sticky",
        "random_diversification",
        "mixed_min_pressure",
    }
    assert all(row.p50_resume_s >= 0 for row in rows)
    summary = summarize_cells(rows)
    assert len(summary) == 1
    assert summary[0]["best_policy"] in {row.policy for row in rows}
    assert "mixed_vs_strong_reuse_gap_frac" in summary[0]


def test_k8_budget_applies_link_and_prefill_axes():
    """Claim: the K8 axes become ResourceBudget values consumed by K4."""
    cell = RegimeCell(
        n_workflows=10,
        state_scale="tiny",
        prefill_capacity="moderate",
        link_gbps=25,
    )
    budget = make_k8_budget(cell)
    assert budget.prefill_tok_s_per_site["seattle"] == 100_000.0
    assert budget.network_bps_per_link[("phoenix", "seattle")] == 25e9


def test_k8_link_axis_does_not_change_workload_shape():
    """Claim: changing link bandwidth changes only the resource budget, not
    the synthetic workload being evaluated."""
    slow = RegimeCell(
        n_workflows=10,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=1,
        seed=8120,
    )
    fast = RegimeCell(
        n_workflows=10,
        state_scale="medium",
        prefill_capacity="tight",
        link_gbps=100,
        seed=8120,
    )
    _, slow_manifests = make_k8_episode(slow)
    _, fast_manifests = make_k8_episode(fast)

    for workflow_id in slow_manifests:
        slow_states = slow_manifests[workflow_id].state_objects
        fast_states = fast_manifests[workflow_id].state_objects
        assert slow_states.keys() == fast_states.keys()
        for state_id in slow_states:
            assert slow_states[state_id].tokens == fast_states[state_id].tokens
            assert slow_states[state_id].bytes == fast_states[state_id].bytes



def test_k8_exact_vs_aggregate_calibration_preserves_cell_identity(tmp_path):
    """Claim: exact_vs_aggregate calibration compares the same semantic cell
    for every policy and writes enough metadata to audit disagreement."""
    bundle = default_bundle(REPO)
    rows = calibrate_k8_estimator(
        bundle,
        n_values=(10,),
        state_scales=("tiny",),
        prefill_caps=("tight",),
        link_gbps_values=(1,),
    )
    assert len(rows) == 6
    assert {row.cell.cell_id for row in rows} == {"n10_tiny_tight_1g"}
    assert {row.policy for row in rows} == {
        "strong_reuse",
        "replay_all",
        "kv_all",
        "workspace_sticky",
        "random_diversification",
        "mixed_min_pressure",
    }
    assert all(row.relative_p50_error >= 0 for row in rows)

    write_k8_calibration_artifacts(rows, tmp_path)
    text = (tmp_path / "exact_vs_aggregate.csv").read_text()
    assert "relative_p50_error" in text
    assert "n10_tiny_tight_1g" in text
