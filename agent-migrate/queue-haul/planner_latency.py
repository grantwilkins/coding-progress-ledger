"""Selection latency for the pure LP and the pure greedy against fleet size.

Times only the selection step.  A whole ``plan`` call is dominated by work both
policies share: at 50,000 sessions it costs about 40 s, of which selection is
4.5 s, while the single simulation run to check the plan against its deadline
takes 23.5 s, the candidate table 5.8 s and replica packing 4.1 s.  An
end-to-end timing therefore hides the solver difference and can invert it -- the
LP measures 38.7 s against the greedy's 42.4 s purely because it selected 192
fewer moves for that shared simulation to execute.

Both solvers are held to their pure form.  The greedy runs without its integral
repair, so it never reaches for the exact MILP, and the target is one both can
attain, so the LP runs its target-first solve rather than the max-shed fallback
it shares with the greedy on an impossible target.  Each selection must reach
the target or the run fails: a policy scored in its fallback is not the policy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import fleet_shed_frontier_campaign as campaign
import pool_planner
from planner import source_power
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile

SESSIONS = (28, 1_000, 10_000, 50_000, 100_000)
FRACTION = 0.25
RHO = 0.38
DEADLINE_S = 600.0
REPEATS = 3


def _selection_credit(table, selected):
    return sum(table.candidates[i].credit for i in selected)


def measure(sessions: int, repeats: int = REPEATS) -> dict:
    profile = ModelProfile.load(campaign.MODEL)
    workload = WorkloadProfile.load(
        campaign.WORKLOADS[campaign.HEADLINE_WORKLOAD])
    bound = campaign.request_work(profile.case()).sum() * 5.0
    bounds = {mode: bound for mode in ("normal", "emergency", "stable")}
    scenario, replicas, demand, fits = campaign.build_fleet(
        profile, workload, sessions, 1001, DEADLINE_S, bound, "natural")
    architecture = campaign.build_architecture(
        profile, replicas, bounds, fits, RHO,
        campaign.migration_headroom(RHO, demand, replicas, bound), None)
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    target = FRACTION * (initial - source_power(
        scenario, profile, [s.session_id for s in scenario.sessions]))
    started = perf_counter()
    table = pool_planner.candidate_table(
        replace(scenario, power_limit_w=initial - target), profile,
        architecture, "normal", power)
    build_s = perf_counter() - started

    row = {"sessions": sessions, "source_replicas": replicas,
           "candidates": len(table.candidates), "target_w": target,
           "table_build_s": build_s}
    for name, select in (("lp", lambda: pool_planner._lp(table, target)),
                         ("greedy", lambda: pool_planner._greedy(table, target))):
        times = []
        for _ in range(repeats):
            started = perf_counter()
            selected = select()
            times.append(perf_counter() - started)
        credit = _selection_credit(table, selected)
        if credit < target - 1e-6:
            raise RuntimeError(
                f"{name} reached {credit:.1f} W of a {target:.1f} W target at "
                f"{sessions} sessions: it answered from a fallback, so the "
                f"timing is not this policy's. Lower FRACTION.")
        row[f"{name}_select_s"] = statistics.median(times)
        row[f"{name}_moves"] = len(selected)
    row["lp_over_greedy"] = row["lp_select_s"] / row["greedy_select_s"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, nargs="+", default=SESSIONS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", type=Path,
                        default=campaign.ROOT / "outputs/planner-latency")
    args = parser.parse_args()
    rows = []
    for sessions in args.sessions:
        rows.append(measure(sessions, args.repeats))
        print(f"{rows[-1]['sessions']:>7} sessions "
              f"({rows[-1]['candidates']:>7} candidates): "
              f"lp {rows[-1]['lp_select_s']:8.3f}s  "
              f"greedy {rows[-1]['greedy_select_s']:8.3f}s  "
              f"({rows[-1]['lp_over_greedy']:.1f}x)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "planner_latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
