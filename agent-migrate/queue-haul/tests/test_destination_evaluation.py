"""
Claim:
Measurement reduction uses independent runs and conservative one-sided choices,
while the architecture grid computes the stated facet headroom and integer replica
allocation deterministically.

Plausible wrong implementations:
- Treat requests as independent repeats or take a mean as the conservative bound.
- Treat a censored or infeasible inner point as measured capacity.
- Clamp a changed runtime baseline into the load-induced slowdown.
- Choose the median rather than worst observed migration slowdown.
- Divide total capacity rather than residual capacity by reference demand.
- Round replica demand down or permit fewer replicas than pools.
- Change the preregistered rho/headroom/pool grid order or contents.
"""

from types import SimpleNamespace

import pytest

import destination_evaluation
from destination_evaluation import (SweepCell, effective_headroom, primary_cells,
                                    reduce_bounds, reduce_loaded, replica_counts, run_sweep)
from test_pool_planner import architecture


def boundary_rows(emergency=2):
    return [
        {"mode": mode, "facet": 0, "run_id": run, "bound": bound + run / 10,
         "outside": bound + run / 10 + .5, "inside_decision": "feasible",
         "outside_decision": "infeasible"}
        for mode, bound in (("normal", 1), ("emergency", emergency), ("stable", 3))
        for run in range(3)
    ]


def test_envelope_reduction_uses_run_median_and_conservative_minimum():
    reduced = reduce_bounds(boundary_rows())

    assert reduced["central"]["normal"] == pytest.approx((1.1,))
    assert reduced["conservative"]["normal"] == (1,)
    with pytest.raises(ValueError, match="nonnested"):
        reduce_bounds(boundary_rows(emergency=.5))


def test_envelope_reduction_rejects_censored_boundary():
    rows = boundary_rows()
    rows[0]["outside"] = rows[0]["bound"]
    rows[0]["inside_decision"] = "infeasible"

    with pytest.raises(ValueError, match="bracketed"):
        reduce_bounds(rows)


def test_loaded_reduction_uses_worst_run_per_load():
    rows = [
        {"method": method, "rho": rho, "run_id": run,
         "duration_factor": .5 * (1 + rho + run / 10),
         "achieved_rho": rho, "context_tokens": 10 + 10 * run,
         "bandwidth_bytes_per_s": 5 + 5 * run}
        for method in ("replay", "kv_transfer") for rho in (0, .5, 1)
        for run in range(3)
    ]
    reduced = reduce_loaded(rows, "hand")

    assert reduced["central"]["replay"].baseline_factor == pytest.approx(.55)
    assert reduced["conservative"]["replay"].baseline_factor == pytest.approx(.6)
    assert reduced["central"]["replay"].slowdown == pytest.approx(
        (1, 1.6 / 1.1, 2.1 / 1.1)
    )
    assert reduced["conservative"]["replay"].slowdown == pytest.approx(
        (1, 1.7 / 1.2, 2.2 / 1.2)
    )


def test_loaded_reduction_rejects_missing_or_missed_baseline():
    rows = [
        {"method": method, "rho": .5, "run_id": run, "duration_factor": 1,
         "achieved_rho": .4, "context_tokens": 10 + 10 * run,
         "bandwidth_bytes_per_s": 5 + 5 * run}
        for method in ("replay", "kv_transfer") for run in range(3)
    ]
    with pytest.raises(ValueError, match="matched unloaded"):
        reduce_loaded(rows, "hand")


def test_effective_headroom_uses_aggregate_residual_facets():
    arch = architecture(normal=.5, emergency=.5, stable=.5,
                        baselines=((.1, 0), (.1, 0)))

    assert effective_headroom(arch, (.2, 0)) == pytest.approx(4)


def test_replica_allocation_rounds_up_and_never_undersupplies_pools():
    q = architecture(normal=.5, emergency=.5, stable=.5).types[0]

    assert replica_counts(q, 4, .5, 1, (.5, 0)) == (1, 1, 1, 1)
    assert replica_counts(q, 4, .5, 3, (.5, 0)) == (2, 2, 1, 1)
    assert sum(replica_counts(q, 4, .5, 4, (.5, 0))) \
        >= sum(replica_counts(q, 4, .5, 3, (.5, 0)))
    assert sum(replica_counts(q, 4, .8, 3, (.5, 0))) \
        >= sum(replica_counts(q, 4, .5, 3, (.5, 0)))


def test_primary_grid_is_deterministic_and_complete():
    assert len(primary_cells()) == 36
    assert primary_cells()[0] == primary_cells()[0].__class__(0, .5, 1)
    assert primary_cells()[-1] == primary_cells()[-1].__class__(.95, 2, 8)


def test_seeded_sweep_repeats_only_transition_cells(monkeypatch):
    calls = []
    def build(cell, seed):
        calls.append((cell, seed))
        return object(), architecture(), {}, (.2, 0)
    def fake_plan(_scenario, _profile, _routes, solver, seed, destination):
        feasible = destination is not None and seed % 2 == 0
        return SimpleNamespace(
            feasible=feasible, initial_source_power_w=2, planned_source_power_w=1,
            power_shortfall_w=0, admission_mode="normal", predicted_migration_makespan_s=1,
            bottleneck="service:p0", packing_repair_count=0, solve_s=0,
            planner_memory_bytes=1,
        )
    monkeypatch.setattr(destination_evaluation, "plan", fake_plan)
    cell = SweepCell(.8, 1, 1)

    first = run_sweep(build, object(), (cell,), range(2), range(2, 4))
    second = run_sweep(build, object(), (cell,), range(2), range(2, 4))

    def key(row):
        return row["seed"], row["planner"], row["feasible"]
    assert list(map(key, first)) == list(map(key, second))
    assert {r["seed"] for r in first} == {0, 1, 2, 3}
