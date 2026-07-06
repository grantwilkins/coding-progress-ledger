"""Claim:
Execution validation recomputes node-knee power on the selected, egress-completed,
and rebuild-completed subsets of a replayed active-knee plan.

Plausible wrong implementations:
- Reuse selected jobs for every realized metric.
- Use active-floor watts where node-expected watts are required.
- Count rebuild-completed jobs in the egress subset or vice versa.
- Drop the ordering dimension from the validation sweep.
- Define the execution target from active-floor watts instead of full node-expected watts.
- Re-solve inside the fixed-plan replay sweep.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import Plan
from impact import Impact
from instance import JobPopulation
from node_knee import evaluate_node_expected_w, execution_realization_metrics
from plot_node_knee_execution_validation import ORDERS, TARGET_FRAC, VARIANTS, run_fixed_plan_sweep, run_sweep
from power import PoolPower
from simulate import SimResult


def _pop():
    z = np.zeros(3)
    ell = np.array([0.10, 0.05, 0.05])
    return JobPopulation(np.array(["chat"] * 3), np.array(["ordinary_chat"] * 3),
                         np.array(["active"] * 3), np.zeros(3, bool), np.full(3, 1e4),
                         z, z, z, z, z, np.ones(3, bool), ell, z, z, "bf16", 0.35,
                         np.array([0, 0, 1]))


def _imp():
    dp = np.array([10.0, 20.0, 30.0])
    z = np.zeros(3)
    return Impact(dp, dp, dp, dp, z, z, z, z, z, "load")


def test_execution_realization_metrics_use_completion_subsets():
    pop, pool, imp = _pop(), PoolPower(), _imp()
    plan = Plan(np.zeros(3), np.ones(3), 0.0, 0.0, 120.0, True, 0.0, "load", "test")
    sim = SimResult(
        np.zeros(3), np.array([1.0, 3.0, 1.0]), np.zeros(3), np.array([5.0, 3.0, 1.0]),
        0.0, 0.0, 0, 5.0, 0.0, 5.0, np.zeros(1), np.zeros(1), np.ones(1), "fifo", "sf"
    )
    got = execution_realization_metrics(pop, pool, imp, plan, sim, D=2.0)

    assert got["selected_node_expected_w"] == pytest.approx(evaluate_node_expected_w(pop, pool, [1, 1, 1]))
    assert got["egress_realized_node_expected_w"] == pytest.approx(evaluate_node_expected_w(pop, pool, [1, 0, 1]))
    assert got["rebuild_realized_node_expected_w"] == pytest.approx(evaluate_node_expected_w(pop, pool, [0, 0, 1]))
    assert got["selected_active_floor_w"] == 60.0
    assert got["egress_realized_active_floor_w"] == 40.0
    assert got["rebuild_realized_active_floor_w"] == 30.0
    assert got["rebuild_realized_node_s_per_kw"] == pytest.approx(
        plan.cost / (got["rebuild_realized_node_expected_w"] / 1e3)
    )


def test_execution_validation_sweep_has_variants_orders_and_realization_levels():
    _, target_kw, rows = run_sweep(deadlines=np.array([10.0]))
    assert len(rows) == len(VARIANTS) * len(ORDERS)
    assert target_kw > 0
    assert {r["variant"] for r in rows} == set(VARIANTS)
    assert {r["ordering"] for r in rows} == set(ORDERS)

    for variant in VARIANTS:
        rs = [r for r in rows if r["variant"] == variant]
        assert len({r["selected_node_kw"] for r in rs}) == 1
        for r in rs:
            assert r["target_basis"] == "full_node_expected"
            assert r["target_kw"] == pytest.approx(TARGET_FRAC * r["full_node_kw"])
            assert r["egress_realized_node_kw"] <= r["selected_node_kw"] + 1e-9
            assert r["rebuild_realized_node_kw"] <= r["egress_realized_node_kw"] + 1e-9
            assert r["selected_over_target"] == pytest.approx(r["selected_node_kw"] / r["target_kw"])
            assert r["egress_realized_over_target"] == pytest.approx(r["egress_realized_node_kw"] / r["target_kw"])
            assert r["rebuild_realized_over_target"] == pytest.approx(r["rebuild_realized_node_kw"] / r["target_kw"])


def test_fixed_plan_replay_is_monotone_and_uses_one_plan_deadline():
    deadlines = np.array([6.0, 10.0, 30.0])
    _, _, rows = run_fixed_plan_sweep(deadlines=deadlines)
    assert {r["sweep"] for r in rows} == {"fixed_plan_replay"}
    for variant in VARIANTS:
        for order in ORDERS:
            rs = [r for r in rows if r["variant"] == variant and r["ordering"] == order]
            assert len({r["plan_deadline_s"] for r in rs}) == 1
            assert len({r["cost_s"] for r in rs}) == 1
            assert len({r["selected_node_kw"] for r in rs}) == 1
            egress = np.array([r["egress_realized_node_kw"] for r in rs])
            rebuild = np.array([r["rebuild_realized_node_kw"] for r in rs])
            assert np.all(egress[1:] >= egress[:-1] - 1e-9)
            assert np.all(rebuild[1:] >= rebuild[:-1] - 1e-9)
