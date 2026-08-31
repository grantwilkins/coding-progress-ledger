"""Compare greedy and LP plans plus the exact aggregate candidate model."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import vstack

import fleet_shed_frontier_campaign as campaign
import pool_planner
from planner import plan, source_power
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile


def exact_selection(table, target, max_gain=None):
    gains = np.asarray([candidate.credit for candidate in table.candidates])
    durations = np.asarray([candidate.duration_s for candidate in table.candidates])
    matrix = vstack((table.incidence, table.resources), format="csr")
    base = LinearConstraint(matrix, -np.inf, 1)
    kwargs = dict(integrality=np.ones(len(gains)), bounds=Bounds(0, 1),
                  options={"mip_rel_gap": 0})
    if max_gain is None:
        maximum = milp(-gains, constraints=base, **kwargs)
        if not maximum.success or maximum.mip_gap > 1e-8:
            raise RuntimeError(f"{maximum.message}; MIP gap={maximum.mip_gap}")
        max_gain = -float(maximum.fun)
    if target <= 1e-8:
        return max_gain, 0.0
    if target > max_gain + 1e-7:
        return max_gain, None
    minimum = milp(durations, constraints=(
        base, LinearConstraint(gains, target - 1e-8, np.inf),
    ), **kwargs)
    if not minimum.success or minimum.mip_gap > 1e-8:
        raise RuntimeError(f"{minimum.message}; MIP gap={minimum.mip_gap}")
    return max_gain, float(minimum.fun)


def build_case(workload_name, sessions, seed, deadline_s, rho):
    profile = ModelProfile.load(campaign.MODEL)
    workload = WorkloadProfile.load(campaign.WORKLOADS[workload_name])
    bound = campaign.request_work(profile.case()).sum() * 5
    bounds = {mode: bound for mode in ("normal", "emergency", "stable")}
    scenario, replicas, demand, fits = campaign.build_fleet(
        profile, workload, sessions, seed, deadline_s, bound, "natural",
    )
    architecture = campaign.build_architecture(
        profile, replicas, bounds, fits, rho,
        campaign.migration_headroom(rho, demand, replicas, bound), None,
    )
    return profile, scenario, architecture


def evaluate_case(workload, sessions, seed, deadline_s, rho, fraction,
                  target_basis="exact"):
    profile, scenario, architecture = build_case(
        workload, sessions, seed, deadline_s, rho,
    )
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    removable = initial - source_power(
        scenario, profile, (session.session_id for session in scenario.sessions),
    )
    sessions_ = tuple(session for session in scenario.sessions
                      if session.state == "active")
    credits, _, _ = pool_planner._phase_power_target(power, sessions_, 0)
    table = pool_planner.candidate_table(
        scenario, profile, architecture, "normal", power, credits,
    )
    exact_max, _ = exact_selection(table, 0)
    ask = fraction * (exact_max if target_basis == "exact" else removable)
    _, selection_target, _ = pool_planner._phase_power_target(power, sessions_, ask)
    exact_max, exact_duration = exact_selection(table, selection_target, exact_max)
    candidates = {
        (table.sessions[candidate.session].session_id, candidate.method,
         architecture.pools[candidate.pool].pool_id): candidate
        for candidate in table.candidates
    }
    row = {
        "workload": workload, "sessions": sessions, "seed": seed,
        "deadline_s": deadline_s, "rho": rho, "target_basis": target_basis,
        "target_fraction": fraction, "target_w": ask,
        "selection_target": selection_target, "exact_max_credit": exact_max,
        "exact_aggregate_feasible": exact_duration is not None,
        "exact_aggregate_min_duration_s": exact_duration,
    }
    for solver in ("lp", "greedy"):
        result = plan(
            replace(scenario, power_limit_w=initial - ask), profile, {}, solver,
            seed=seed, destination=architecture, admission_mode="normal",
        )
        chosen = [candidates[
            move.session_id, move.method, move.destination_pool
        ] for move in result.moves]
        selected_credit = sum(candidate.credit for candidate in chosen)
        duration = sum(candidate.duration_s for candidate in chosen)
        row.update({
            f"{solver}_feasible": result.feasible,
            f"{solver}_power_hit": result.power_shortfall_w <= 1e-8,
            f"{solver}_selection_hit": selected_credit >= selection_target - 1e-7,
            f"{solver}_shed_w": initial - result.planned_source_power_w,
            f"{solver}_target_ratio": (initial - result.planned_source_power_w)
            / max(ask, 1e-12),
            f"{solver}_selected_credit": selected_credit,
            f"{solver}_exact_max_ratio": selected_credit / max(exact_max, 1e-12),
            f"{solver}_duration_s": duration,
            f"{solver}_duration_ratio": None if exact_duration is None
            else duration / max(exact_duration, 1e-12),
            f"{solver}_moves": len(result.moves),
            f"{solver}_packing_repairs": result.packing_repair_count,
            f"{solver}_deadline_repairs": result.deadline_repair_count,
            f"{solver}_milp_recovery_s": result.milp_recovery_s,
        })
    return row


def summarize(rows):
    exact = [row for row in rows if row["exact_aggregate_feasible"]]
    summary = {"cells": len(rows), "exact_aggregate_feasible_cells": len(exact)}
    for solver in ("lp", "greedy"):
        ratios = np.asarray([row[f"{solver}_target_ratio"] for row in rows])
        durations = np.asarray([
            row[f"{solver}_duration_ratio"] for row in exact
            if row[f"{solver}_selection_hit"] and row[f"{solver}_power_hit"]
        ])
        summary[solver] = {
            "selection_hit_rate_on_exact_aggregate_feasible": float(np.mean([
                row[f"{solver}_selection_hit"] for row in exact
            ])),
            "power_hit_rate_on_exact_aggregate_feasible": float(np.mean([
                row[f"{solver}_power_hit"] for row in exact
            ])),
            "power_hit_rate": float(np.mean([
                row[f"{solver}_power_hit"] for row in rows
            ])),
            "feasible_rate": float(np.mean([
                row[f"{solver}_feasible"] for row in rows
            ])),
            "target_ratio_p05_median_p95": list(map(float, np.quantile(
                ratios, (.05, .5, .95),
            ))),
            "duration_ratio_median_p95_max": list(map(float, (
                np.median(durations), np.quantile(durations, .95), durations.max(),
            ))) if len(durations) else [],
            "worst_exact_max_ratio": float(min(
                (row[f"{solver}_exact_max_ratio"] for row in rows
                 if not row["exact_aggregate_feasible"]), default=1,
            )),
            "milp_recovery_cells": sum(
                row[f"{solver}_milp_recovery_s"] > 0 for row in rows
            ),
        }
    summary["lp_hit_greedy_miss"] = sum(
        row["lp_power_hit"] and not row["greedy_power_hit"] for row in rows
    )
    summary["greedy_hit_lp_miss"] = sum(
        row["greedy_power_hit"] and not row["lp_power_hit"] for row in rows
    )
    lp_hits = [row for row in rows if row["lp_power_hit"]]
    both = [row for row in lp_hits if row["greedy_power_hit"]]
    shed_ratios = np.asarray([
        row["greedy_shed_w"] / row["lp_shed_w"] for row in rows
        if row["lp_shed_w"] > 1e-12
    ])
    duration_ratios = np.asarray([
        row["greedy_duration_s"] / row["lp_duration_s"] for row in both
        if row["lp_duration_s"] > 1e-12
    ])
    summary["paired"] = {
        "lp_power_hit_cells": len(lp_hits),
        "greedy_power_hit_given_lp_hit": float(np.mean([
            row["greedy_power_hit"] for row in lp_hits
        ])) if lp_hits else None,
        "greedy_feasible_given_lp_feasible": float(np.mean([
            row["greedy_feasible"] for row in rows if row["lp_feasible"]
        ])) if any(row["lp_feasible"] for row in rows) else None,
        "greedy_over_lp_shed_p05_median_p95": list(map(float, np.quantile(
            shed_ratios, (.05, .5, .95),
        ))),
        "greedy_over_lp_duration_median_p95": list(map(float, (
            np.median(duration_ratios), np.quantile(duration_ratios, .95),
        ))) if len(duration_ratios) else [],
    }
    fields = ("workload", "sessions", "seed", "deadline_s", "rho",
              "target_basis", "target_fraction")
    summary["worst_cells"] = {}
    for solver in ("lp", "greedy"):
        values = [row for row in exact if row[f"{solver}_selection_hit"]
                  and row[f"{solver}_power_hit"]]
        if values:
            worst = max(values, key=lambda row: row[f"{solver}_duration_ratio"])
            summary["worst_cells"][f"{solver}_duration"] = {
                **{field: worst[field] for field in fields},
                "duration_ratio": worst[f"{solver}_duration_ratio"],
            }
    misses = [row for row in rows
              if row["lp_power_hit"] and not row["greedy_power_hit"]]
    if misses:
        worst = min(misses, key=lambda row: row["greedy_target_ratio"])
        summary["worst_cells"]["lp_hit_greedy_miss"] = {
            **{field: worst[field] for field in fields},
            "greedy_target_ratio": worst["greedy_target_ratio"],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", choices=tuple(campaign.WORKLOADS),
                        default=(campaign.HEADLINE_WORKLOAD,))
    parser.add_argument("--sessions", type=int, nargs="+", default=(8, 32, 128))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--deadlines", type=float, nargs="+", default=(60, 300))
    parser.add_argument("--rhos", type=float, nargs="+", default=(.2, .38, .48))
    parser.add_argument("--targets", type=float, nargs="+",
                        default=(.1, .25, .5, .75, .9, 1, 1.05))
    parser.add_argument("--target-basis", choices=("exact", "removable"),
                        default="exact")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [evaluate_case(workload, n, 1001 + seed, deadline, rho, target,
                          args.target_basis)
            for workload in args.workloads for n in args.sessions
            for seed in range(args.seeds)
            for deadline in args.deadlines for rho in args.rhos
            for target in args.targets]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
