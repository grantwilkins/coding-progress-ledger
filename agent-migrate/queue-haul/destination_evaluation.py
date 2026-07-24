"""Conservative measurement reduction and compact architecture sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from destination import LoadedCoefficients
from planner import plan


RHO = (0, .5, .8, .95)
HEADROOM = (.5, 1, 2)
POOLS = (1, 4, 8)


def archived_cache_state(requests, block_tokens):
    """Audit legacy cache geometry without claiming complete service evidence."""
    return _cache_state(requests, block_tokens, strict=False)


def service_cache_state(requests, block_tokens):
    """Validate current service evidence and its intended warmed-prefix state."""
    return _cache_state(requests, block_tokens, strict=True)


def _cache_state(requests, block_tokens, strict):
    if not requests or block_tokens < 1:
        raise ValueError("cache classification needs requests and block size")
    geometry = ("prompt_tokens", "input_tokens", "cached_tokens")
    completion = (
        "planned_prompt_tokens", "output_tokens", "planned_output_tokens",
    )
    counts = dict.fromkeys(
        ("private_prefix", "prefix_underhit", "append_hot", "measurement_invalid"), 0
    )
    for row in requests:
        if row.get("status") != 200 or row.get("error") \
                or any(key not in row for key in geometry) \
                or strict and (row.get("done") is not True
                               or any(key not in row for key in completion)):
            state = "measurement_invalid"
        else:
            prompt, appended = int(row["prompt_tokens"]), int(row["input_tokens"])
            cached = int(row["cached_tokens"])
            if prompt <= 0 or not 0 < appended <= prompt \
                    or not 0 <= cached <= prompt \
                    or strict and (
                        int(row["planned_prompt_tokens"]) != prompt
                        or int(row["output_tokens"])
                        != int(row["planned_output_tokens"])
                    ):
                state = "measurement_invalid"
            else:
                warmed = (prompt - appended) // block_tokens * block_tokens
                state = ("append_hot" if cached > warmed else
                         "prefix_underhit" if cached < warmed else "private_prefix")
        counts[state] += 1
    state = next(name for name in
                 ("measurement_invalid", "append_hot", "prefix_underhit", "private_prefix")
                 if counts[name])
    return {"state": state, "requests": counts}


def reduce_bounds(rows):
    """Return central medians and conservative run minima by mode and facet."""
    if any(r.get("cache_state") != "private_prefix" for r in rows):
        raise ValueError("service envelope needs exact private-prefix runs")
    if any(
        r.get("inside_decision") != "feasible"
        or r.get("outside_decision") != "infeasible"
        or float(r.get("outside", 0)) <= float(r["bound"])
        for r in rows
    ):
        raise ValueError("service envelope needs bracketed feasible/infeasible cells")
    out = {}
    for conservative, name in ((False, "central"), (True, "conservative")):
        values = {}
        for mode in ("normal", "emergency", "stable"):
            facets = sorted({int(r["facet"]) for r in rows if r["mode"] == mode})
            values[mode] = tuple(
                (min if conservative else np.median)(
                    [float(r["bound"]) for r in rows
                     if r["mode"] == mode and int(r["facet"]) == facet]
                ) for facet in facets
            )
            if not facets or any(len({r["run_id"] for r in rows
                                      if r["mode"] == mode and int(r["facet"]) == facet}) < 3
                                 for facet in facets):
                raise ValueError("each envelope cell needs three independent runs")
        a = np.asarray([values[m] for m in ("normal", "emergency", "stable")])
        if a.ndim != 2 or np.any(np.diff(a, axis=0) < 0):
            raise ValueError("reduced envelopes are nonnested")
        out[name] = values
    return out


def reduce_loaded(rows, provenance):
    """Separate matched-runtime baseline calibration from loaded slowdown."""
    result = {case: {} for case in ("central", "conservative")}
    for method in ("replay", "kv_transfer"):
        selected = [r for r in rows if r["method"] == method]
        rhos = sorted({float(r["rho"]) for r in selected})
        if not rhos or rhos[0] != 0 or len(rhos) < 2 or any(
            len({r["run_id"] for r in selected if float(r["rho"]) == rho}) < 3
            for rho in rhos
        ) or any(
            "duration_factor" not in r or "achieved_rho" not in r
            or float(r.get("duration_factor", 0)) <= 0
            or abs(float(r.get("achieved_rho", math.inf)) - float(r["rho"])) > .05
            for r in selected
        ):
            raise ValueError(
                "loaded migration needs matched unloaded and achieved-load runs"
            )
        common = (tuple(rhos), None,
                  (min(float(r["context_tokens"]) for r in selected),
                   max(float(r["context_tokens"]) for r in selected)),
                  (min(float(r["bandwidth_bytes_per_s"]) for r in selected),
                   max(float(r["bandwidth_bytes_per_s"]) for r in selected)), provenance)
        if common[2][0] == common[2][1] or common[3][0] == common[3][1]:
            raise ValueError("loaded profile needs context and bandwidth ranges")
        for case, reducer in (("central", np.median), ("conservative", max)):
            factors = tuple(float(reducer([
                float(r["duration_factor"]) for r in selected
                if float(r["rho"]) == rho
            ])) for rho in rhos)
            baseline = factors[0]
            slowdowns = tuple(max(1.0, value / baseline) for value in factors)
            result[case][method] = LoadedCoefficients(
                common[0], slowdowns, *common[2:], baseline
            )
    return result


def effective_headroom(architecture, reference_demand) -> float:
    """Ideal aggregate residual normal capacity divided by reference demand."""
    types = architecture.type_by_id
    normals = np.asarray(types[architecture.pools[0].type_id].normals)
    if any(not np.array_equal(normals, q.normals) for q in types.values()):
        raise ValueError("headroom requires common facet normals")
    residual = np.zeros(len(normals))
    for pool in architecture.pools:
        q = types[pool.type_id]
        baseline = sum((np.asarray(r.baseline_work) for r in pool.replicas), start=np.zeros(2))
        residual += len(pool.replicas) * np.asarray(q.bounds["normal"]) - normals @ baseline
    demand = normals @ np.asarray(reference_demand)
    return float(min(residual[demand > 0] / demand[demand > 0]))


def replica_counts(pool_type, pool_count, rho, headroom, reference_demand):
    """Smallest balanced integer replica allocation meeting requested ideal headroom."""
    if not 0 <= rho < 1 or pool_count < 1 or headroom <= 0:
        raise ValueError("invalid sweep cell")
    normals, bounds = np.asarray(pool_type.normals), np.asarray(pool_type.bounds["normal"])
    total = max(pool_count, math.ceil(max(
        headroom * normals @ np.asarray(reference_demand) / ((1 - rho) * bounds)
    )))
    return tuple(total // pool_count + (i < total % pool_count) for i in range(pool_count))


@dataclass(frozen=True)
class SweepCell:
    rho: float
    headroom: float
    pools: int


def primary_cells(): return tuple(SweepCell(*x) for x in product(RHO, HEADROOM, POOLS))


def run_sweep(build, profile, cells=primary_cells(), seeds=range(10),
              transition_seeds=range(10, 30)):
    """Run paired legacy, pool-LP, and pool-greedy cells from a seeded builder."""
    rows, cells, seeds = [], tuple(cells), tuple(seeds)

    def run(selected_cells, selected_seeds):
        for cell, seed in product(selected_cells, selected_seeds):
            scenario, architecture, routes, *reference = build(cell, seed)
            for label, solver, destination in (
                ("scalar", "lp", None), ("pool_lp", "lp", architecture),
                ("pool_greedy", "greedy", architecture),
            ):
                result = plan(
                    scenario, profile, routes, solver, seed=seed, destination=destination,
                )
                rows.append({
                "rho": cell.rho, "headroom": cell.headroom, "pools": cell.pools,
                "seed": seed, "planner": label, "feasible": result.feasible,
                "shed_w": result.initial_source_power_w - result.planned_source_power_w,
                "shortfall_w": result.power_shortfall_w, "mode": result.admission_mode,
                "migration_s": result.predicted_migration_makespan_s,
                "bottleneck": result.bottleneck, "packing_repairs": result.packing_repair_count,
                "runtime_s": result.solve_s, "memory_bytes": result.planner_memory_bytes,
                "achieved_headroom": effective_headroom(architecture, reference[0])
                if reference else None,
                })
    run(cells, seeds)
    transitions = [cell for cell in cells if 0 < sum(
        r["feasible"] for r in rows if r["planner"] == "pool_lp"
        and (r["rho"], r["headroom"], r["pools"]) ==
        (cell.rho, cell.headroom, cell.pools)
    ) < len(tuple(seeds))]
    run(transitions, transition_seeds)
    return rows
