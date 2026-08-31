"""
Claim:
Measurement reduction uses independent runs and conservative one-sided choices,
while the architecture grid computes the stated facet headroom and integer replica
allocation deterministically.

Plausible wrong implementations:
- Treat requests as independent repeats or take a mean as the conservative bound.
- Treat a censored or infeasible inner point as measured capacity.
- Compare cache hits with the full prompt instead of the intentionally warmed prefix.
- Accept a run when only a majority of its requests avoid future-append cache hits.
- Treat a zero-work response as a cache-state result instead of measurement-invalid.
- Trust reported token counts after a truncated stream or prompt-usage mismatch.
- Conflate forensic legacy cache geometry with strict future service evidence.
- Ignore physical cache-block rounding at the prefix/append boundary.
- Clamp a changed runtime baseline into the load-induced slowdown.
- Choose the median rather than worst observed migration slowdown.
- Divide total capacity rather than residual capacity by reference demand.
- Round replica demand down or permit fewer replicas than pools.
- Expand every operating point into an impractical Cartesian product.
- Omit the two-pool or fixed-total versus fixed-per-pool cases.
- Conflate normal headroom, event flex, and debt.
"""

from types import SimpleNamespace

import pytest

import destination_evaluation
from destination_evaluation import (SweepCell, archived_cache_state,
                                    effective_headroom, primary_cells,
                                    reduce_bounds, reduce_loaded, replica_counts,
                                    run_sweep, service_cache_state)
from migration import ORDERED_EAGER_PARALLEL_V1
from test_pool_planner import architecture


def boundary_rows(emergency=2):
    return [
        {"mode": mode, "facet": 0, "run_id": run, "bound": bound + run / 10,
         "outside": bound + run / 10 + .5, "inside_decision": "feasible",
         "outside_decision": "infeasible", "cache_state": "private_prefix"}
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


@pytest.mark.parametrize("state", ["append_hot", "prefix_underhit", None])
def test_envelope_reduction_requires_exact_private_prefix_runs(state):
    rows = boundary_rows()
    rows[0]["cache_state"] = state

    with pytest.raises(ValueError, match="private-prefix"):
        reduce_bounds(rows)


def cache_request(cached=96, prompt=113, appended=17, output=1):
    return {
        "status": 200, "error": "", "prompt_tokens": prompt,
        "cached_tokens": cached, "input_tokens": appended,
        "output_tokens": output, "planned_output_tokens": 1,
        "done": True, "planned_prompt_tokens": prompt,
    }


def test_service_cache_state_uses_warmed_prefix_block_boundary():
    assert service_cache_state([cache_request()], 16)["state"] == "private_prefix"
    assert service_cache_state([cache_request(cached=112)], 16)["state"] == "append_hot"
    assert service_cache_state([cache_request(cached=80)], 16)["state"] == "prefix_underhit"


def test_service_cache_state_rejects_one_hot_append_at_run_level():
    result = service_cache_state([cache_request(), cache_request(cached=112)], 16)

    assert result["state"] == "append_hot"
    assert result["requests"] == {
        "private_prefix": 1, "prefix_underhit": 0, "append_hot": 1,
        "measurement_invalid": 0,
    }


def test_service_cache_state_keeps_invalid_work_separate():
    result = service_cache_state([cache_request(cached=112, output=0)], 16)

    assert result["state"] == "measurement_invalid"


@pytest.mark.parametrize(
    "changed", [{"done": False}, {"planned_prompt_tokens": 114}]
)
def test_service_cache_state_requires_complete_exact_prompt_work(changed):
    row = cache_request()
    row.update(changed)

    assert service_cache_state([row], 16)["state"] == "measurement_invalid"


@pytest.mark.parametrize("missing", ["done", "planned_prompt_tokens"])
def test_service_cache_state_requires_completion_evidence(missing):
    row = cache_request()
    row.pop(missing)

    assert service_cache_state([row], 16)["state"] == "measurement_invalid"


def test_archived_cache_state_is_geometry_not_completion_evidence():
    row = cache_request()
    row.pop("done")
    row.pop("planned_prompt_tokens")

    assert archived_cache_state([row], 16)["state"] == "private_prefix"
    assert service_cache_state([row], 16)["state"] == "measurement_invalid"


def test_service_cache_state_classifies_http_failure_without_usage():
    row = {"status": 503, "error": "unavailable", "done": False}

    assert service_cache_state([row], 16)["state"] == "measurement_invalid"


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
    assert len(primary_cells()) == 19
    assert primary_cells()[0] == SweepCell(0, 1, 1, .1, .1)
    assert primary_cells()[-1] == SweepCell(.8, 1, 1, .1, .20)
    assert {cell.pools for cell in primary_cells()} == {1, 2, 4, 8}
    assert {
        cell.budget_policy for cell in primary_cells() if cell.pools > 1
    } == {"fixed_total", "fixed_per_pool"}


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
                planner_memory_bytes=1, failure_reason=None,
                candidate_generation_s=0, selection_s=0, milp_recovery_s=0,
                packing_s=0, validation_s=0,
            )
    monkeypatch.setattr(destination_evaluation, "plan", fake_plan)
    cell = SweepCell(.8, 1, 1, .1, .05)

    first = run_sweep(build, object(), (cell,), range(2), range(2, 4))
    second = run_sweep(build, object(), (cell,), range(2), range(2, 4))

    def key(row):
        return row["seed"], row["planner"], row["feasible"]
    assert list(map(key, first)) == list(map(key, second))
    assert {r["seed"] for r in first} == {0, 1, 2, 3}
    assert {r["planner"] for r in first} == {
        "scalar", "pool_lp", "pool_greedy", "pool_lagrangian",
    }
    assert {(r["flex"], r["debt"]) for r in first} == {(.1, .05)}
    assert {r["execution_contract"] for r in first} == {ORDERED_EAGER_PARALLEL_V1}
    assert all(r["shed_w"] == (1 if r["feasible"] else 0) for r in first)
    assert {r["selected_shed_w"] for r in first} == {1}
