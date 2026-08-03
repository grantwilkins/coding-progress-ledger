"""
Claim:
The paired experiment gives every solver the same power target and reports shed
as validated only when the complete destination plan is feasible. Its work gap
uses a fractional source-chord LP lower bound.

Plausible wrong implementations:
- Target a fraction of initial power instead of removable power.
- Credit selected shed after packing or temporal validation fails.
- Mix replay and KV occupancy units with unrelated resource rows.
- Give every session its source's full chord gain instead of its load share.
- Reverse the LP target inequality or omit the per-session exclusivity rows.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from greedy_lagrangian_experiment import (
    chord_candidate_gains,
    experiment_stack_hash,
    fractional_chord_work_bound,
    parse_solvers,
    power_limit,
    result_row,
    rss_bytes,
)


def test_chord_gain_is_source_endpoint_gain_split_by_session_load():
    sessions = (
        SimpleNamespace(session_id="a0", source_instance="a"),
        SimpleNamespace(session_id="a1", source_instance="a"),
        SimpleNamespace(session_id="b0", source_instance="b"),
    )
    table = SimpleNamespace(
        sessions=sessions,
        candidates=tuple(SimpleNamespace(session=i) for i in (0, 0, 1, 2)),
    )
    power = SimpleNamespace(
        profile=SimpleNamespace(power_scope="gpu"),
        ell={"a0": 1, "a1": 3, "b0": 2},
        drain_gain=lambda ids: {frozenset(("a0", "a1")): 8,
                                frozenset(("b0",)): 4}[frozenset(ids)],
    )

    assert chord_candidate_gains(table, power) == pytest.approx((2, 2, 6, 4))


def test_fractional_chord_bound_enforces_target_and_session_rows():
    sessions = (
        SimpleNamespace(session_id="a", source_instance="a"),
        SimpleNamespace(session_id="b", source_instance="b"),
    )
    table = SimpleNamespace(
        sessions=sessions,
        candidates=(SimpleNamespace(session=0, migration_work_s=2),
                    SimpleNamespace(session=1, migration_work_s=6)),
        incidence=csr_matrix(np.eye(2)), resources=csr_matrix((0, 2)),
    )
    power = SimpleNamespace(
        profile=SimpleNamespace(power_scope="gpu"),
        ell={"a": 1, "b": 1},
        drain_gain=lambda ids: 2 if frozenset(ids) == {"a"} else 6,
    )

    assert fractional_chord_work_bound(table, 3, power) == pytest.approx(3)

    power.profile.power_scope = "server"
    with pytest.raises(ValueError, match="GPU-scoped"):
        chord_candidate_gains(table, power)


def test_power_target_is_fraction_of_removable_power():
    assert power_limit(100, 40, 0) == 100
    assert power_limit(100, 40, .25) == 85
    assert power_limit(100, 40, 1) == 40
    with pytest.raises(ValueError):
        power_limit(40, 100, .5)


def test_solver_surface_keeps_only_the_two_supported_greedies():
    assert parse_solvers("greedy,greedy_lagrangian") == (
        "greedy", "greedy_lagrangian",
    )
    with pytest.raises(ValueError):
        parse_solvers("lp")
    for retired in ("greedy_bundle", "greedy_coupled", "greedy_prefix"):
        with pytest.raises(ValueError):
            parse_solvers(retired)


def test_experiment_stack_hash_covers_every_python_source(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("one")
    second.write_text("two")
    original = experiment_stack_hash(tmp_path)
    second.write_text("changed")
    assert experiment_stack_hash(tmp_path) != original


def test_peak_rss_units_match_operating_system_contract():
    assert rss_bytes(2, "darwin") == 2
    assert rss_bytes(2, "linux") == 2048


def test_failed_plan_has_zero_validated_shed_and_method_occupancy_only():
    result = SimpleNamespace(
        resource_uses=(
            SimpleNamespace(name="route:wan", used=100, capacity=200,
                            utilization=.5),
            SimpleNamespace(name="migration:p:replay", used=3, capacity=10,
                            utilization=.3),
            SimpleNamespace(name="migration:p:kv_transfer", used=5, capacity=10,
                            utilization=.5),
        ), initial_source_power_w=100, planned_source_power_w=70,
        expected_source_power_at_deadline_w=75,
        solver="greedy_lagrangian", seed=3, feasible=False, power_shortfall_w=2,
        moves=(SimpleNamespace(method="replay"), SimpleNamespace(method="kv_transfer")),
        solve_s=1, planner_memory_bytes=10, predicted_migration_makespan_s=9,
        packing_repair_count=1, failure_reason="migration_deadline",
        binding_resources=("route:wan",), service_debt_replica_s=2,
        required_recovery_s=4,
    )

    row = result_row(result, "coding", 2, .5, "same-case",
                     requested_shed_w=20)

    assert row["selected_shed_w"] == 30
    assert row["validated_shed_w"] == 0
    assert row["migration_work_s"] == 8
    assert row["unmet_shed_w"] == 0
    assert (row["replay_moves"], row["kv_moves"]) == (1, 1)

    result.feasible = True
    assert result_row(result, "coding", 2, .5, "same-case")[
        "validated_shed_w"
    ] == 25
