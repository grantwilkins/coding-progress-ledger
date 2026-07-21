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


def reduce_bounds(rows):
    """Return central medians and conservative run minima by mode and facet."""
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
    """Return median central and observed-worst conservative slowdown curves."""
    result = {case: {} for case in ("central", "conservative")}
    for method in ("replay", "kv_transfer"):
        selected = [r for r in rows if r["method"] == method]
        rhos = sorted({float(r["rho"]) for r in selected})
        if len(rhos) < 2 or any(len({r["run_id"] for r in selected if float(r["rho"]) == rho}) < 3
                              for rho in rhos):
            raise ValueError("each loaded-migration cell needs three independent runs")
        common = (tuple(rhos), None,
                  (min(float(r["context_tokens"]) for r in selected),
                   max(float(r["context_tokens"]) for r in selected)),
                  (min(float(r["bandwidth_bytes_per_s"]) for r in selected),
                   max(float(r["bandwidth_bytes_per_s"]) for r in selected)), provenance)
        if common[2][0] == common[2][1] or common[3][0] == common[3][1]:
            raise ValueError("loaded profile needs context and bandwidth ranges")
        for case, reducer in (("central", np.median), ("conservative", max)):
            slowdowns = tuple(float(reducer([float(r["slowdown"]) for r in selected
                                             if float(r["rho"]) == rho])) for rho in rhos)
            result[case][method] = LoadedCoefficients(common[0], slowdowns, *common[2:])
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
    rows = []

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
