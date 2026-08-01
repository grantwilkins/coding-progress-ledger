"""
Claim:
The paired experiment gives every solver the same power target and reports shed
as validated only when the complete destination plan is feasible.

Plausible wrong implementations:
- Target a fraction of initial power instead of removable power.
- Credit selected shed after packing or temporal validation fails.
- Mix replay and KV occupancy units with unrelated resource rows.
"""

from types import SimpleNamespace

import pytest

from dual_lagrangian_experiment import (
    experiment_stack_hash,
    parse_solvers,
    power_limit,
    result_row,
)


def test_power_target_is_fraction_of_removable_power():
    assert power_limit(100, 40, 0) == 100
    assert power_limit(100, 40, .25) == 85
    assert power_limit(100, 40, 1) == 40
    with pytest.raises(ValueError):
        power_limit(40, 100, .5)


def test_scale_solver_subset_rejects_nonexperimental_policy():
    assert parse_solvers("greedy,greedy_prefix") == ("greedy", "greedy_prefix")
    with pytest.raises(ValueError):
        parse_solvers("lp")


def test_experiment_stack_hash_covers_every_python_source(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("one")
    second.write_text("two")
    original = experiment_stack_hash(tmp_path)
    second.write_text("changed")
    assert experiment_stack_hash(tmp_path) != original


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
        solver="greedy_prefix", seed=3, feasible=False, power_shortfall_w=2,
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
