from __future__ import annotations

"""
Claim:
V1 exact validation compares the same K8 cell under exact K4 and the
aggregate estimator, then classifies whether aggregate labels/timing are
usable for claims.

Plausible wrong implementations:
- best_policy agreement is computed per policy row instead of once per cell;
- p95 timing error is omitted or accidentally aliases p50/p90;
- policy_winner disagreements near ties are treated as strong contradictions;
- validation silently drops policies and still reports a trustworthy cell;
- selected validation artifacts omit the axis metadata needed to audit a cell.
"""

from pathlib import Path

from agent_migrate_agent.fluid_sim import SimulationResult
from agent_migrate_agent.k8_regime import K8_POLICIES, PolicyMetric, RegimeCell, default_bundle
from agent_migrate_agent.k8_validation import (
    ValidationTarget,
    compare_policy_metrics,
    run_k8_validation,
    summarize_validation,
    write_k8_validation_artifacts,
)


REPO = Path(__file__).resolve().parent.parent


def _metric(cell: RegimeCell, policy: str, p50: float, bottleneck: str) -> PolicyMetric:
    return PolicyMetric(
        cell=cell,
        policy=policy,
        p50_resume_s=p50,
        p90_resume_s=1.8 * p50,
        p95_resume_s=1.9 * p50,
        makespan_s=2.0 * p50,
        dominant_bottleneck=bottleneck,
    )


def _policy_map(cell: RegimeCell, values: dict[str, tuple[float, str]]) -> dict[str, PolicyMetric]:
    fallback = max((v[0] for v in values.values()), default=10.0) + 10.0
    return {
        policy: _metric(cell, policy, *values.get(policy, (fallback, "network")))
        for policy in K8_POLICIES
    }


def test_v1_validation_runs_exact_and_aggregate_for_same_cell():
    """Claim: V1 rows carry p50 and p95 exact_vs_aggregate errors for the
    fixed policy set on the same semantic cell."""
    bundle = default_bundle(REPO)
    target = ValidationTarget(
        "smoke",
        RegimeCell(10, "tiny", "tight", 25, seed=8111),
        "small exact validation smoke cell",
    )
    rows = run_k8_validation(bundle, targets=(target,))
    assert {row.policy for row in rows} == set(K8_POLICIES)
    assert {row.target.cell.cell_id for row in rows} == {"n10_tiny_tight_25g"}
    assert all(row.exact_p95_resume_s >= row.exact_p50_resume_s for row in rows)
    assert all(row.aggregate_p95_resume_s >= row.aggregate_p50_resume_s for row in rows)
    assert all(row.p50_relative_error >= 0.0 for row in rows)
    assert all(row.p95_relative_error >= 0.0 for row in rows)


def test_v1_best_policy_and_trust_labels_are_cell_level():
    """Hand_checkable case: exact winner differs from aggregate winner by
    only 4%, so the cell is a policy boundary rather than a misleading
    aggregate failure."""
    cell = RegimeCell(10, "medium", "tight", 5, seed=8112)
    target = ValidationTarget("near_tie", cell, "hand_worked policy boundary")
    exact = _policy_map(cell, {
        "strong_reuse": (100.0, "prefill"),
        "mixed_min_pressure": (104.0, "prefill"),
    })
    aggregate = _policy_map(cell, {
        "strong_reuse": (110.0, "prefill"),
        "mixed_min_pressure": (90.0, "prefill"),
    })

    rows = compare_policy_metrics(target, exact, aggregate)
    summary = summarize_validation(rows)[0]

    assert summary["exact_best_policy"] == "strong_reuse"
    assert summary["aggregate_best_policy"] == "mixed_min_pressure"
    assert summary["best_policy_agrees"] is False
    assert summary["cell_bottleneck_agrees"] is True
    assert summary["exact_winner_margin_frac"] == 0.04 / 1.04
    assert summary["trust_label"] == "policy_boundary"


def test_v1_validation_refuses_missing_policy():
    """A validation comparison that silently omits a policy can make both
    best_policy and bottleneck agreement look better than they are."""
    cell = RegimeCell(10, "tiny", "loose", 1, seed=8113)
    target = ValidationTarget("bad", cell, "missing policy")
    exact = _policy_map(cell, {})
    aggregate = _policy_map(cell, {})
    aggregate.pop("random_diversification")

    try:
        compare_policy_metrics(target, exact, aggregate)
    except ValueError as exc:
        assert "fixed K8 policy set" in str(exc)
    else:
        raise AssertionError("expected missing_policy validation to fail")


def test_v1_artifacts_include_claim_cell_axes(tmp_path):
    cell = RegimeCell(10, "tiny", "loose", 1, seed=8114)
    target = ValidationTarget("artifact_cell", cell, "artifact metadata")
    exact = _policy_map(cell, {"strong_reuse": (1.0, "network")})
    aggregate = _policy_map(cell, {"strong_reuse": (1.1, "network")})
    rows = compare_policy_metrics(target, exact, aggregate)

    write_k8_validation_artifacts(rows, tmp_path)
    text = (tmp_path / "claim_cell_policy_validation.csv").read_text()

    assert "artifact_cell" in text
    assert "n10_tiny_loose_1g" in text
    assert "exact_p95_resume_s" in text
    assert "p95_relative_error" in text


def test_exact_resume_percentiles_are_completion_thresholds():
    """Hand_checkable case: with ten workflows, p50 is the 5th completion
    and p95 is the 10th completion under ceil(q*N)-1 indexing."""
    result = SimulationResult(
        actions=(),
        final_warmness=None,  # type: ignore[arg_type]
        makespan_s=10.0,
        per_workflow_finish_s={f"wf_{i}": float(i) for i in range(1, 11)},
    )

    assert result.p50_resume_s() == 5.0
    assert result.p90_resume_s() == 9.0
    assert result.p95_resume_s() == 10.0
