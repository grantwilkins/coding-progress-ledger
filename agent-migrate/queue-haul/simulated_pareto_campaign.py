"""Simulate the calibrated width-8 plan and plot its power–completion frontier."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import migration_profiler as profiler
from policy_hardware_campaign import (
    LABELS, _portable_path, _problem, deadline_attainment,
    validate_policy_plan,
)
from planner import plan
from profiles import ActionPower, ModelProfile, RateCurve
from simulate import PlannedMove, execute


ROOT = Path(__file__).parent
DEFAULT_PLAN = ROOT / "outputs/policy-hardware-width8-packing-plan/plan.json"
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
DEFAULT_CROSSOVER = ROOT / "outputs/policy-hardware-crossover-20260730/plan.json"
DEFAULT_WIDTH8 = ROOT / "outputs/policy-hardware-width8-frontier-20260730"
DEFAULT_OUT = ROOT / "outputs/simulated-width8-pareto-20260730"
POLICIES = ("queue_haul", "greedy", "random", "kv_only", "replay_only")
OBSERVATION_S = 600
TIME_BUDGETS_S = (30, 40, 50, 60, 75)


def context_evidence(tokens, anchors):
    values = set(tokens)
    if values <= anchors:
        return "measured"
    if min(values) >= min(anchors) and max(values) <= max(anchors):
        return "interpolated"
    return "extrapolated"


def pareto_flags(rows, keys):
    for row in rows:
        peers = [other for other in rows
                 if all(other[key] == row[key] for key in keys)]
        row["pareto"] = not any(
            other["power_attainment_fraction"]
            >= row["power_attainment_fraction"]
            and other["completion_s"] <= row["completion_s"]
            and (
                other["power_attainment_fraction"]
                > row["power_attainment_fraction"]
                or other["completion_s"] < row["completion_s"]
            )
            for other in peers
        )


def meets_deadline(attainment, completion, deadline):
    return attainment >= 1 - 1e-9 and completion <= deadline + 1e-9


def measured_replay_caps(width8):
    plan_ = json.loads((width8 / "plan.json").read_text())
    totals = {
        row["episode"]: sum(session["initial_tokens"]
                            for session in row["sessions"])
        for row in plan_["scenarios"] if row["policy"] == "control"
    }
    rows = list(csv.DictReader((width8 / "policy_episodes.csv").open()))
    rates = [
        totals[int(row["episode"])] / float(row["commit_100_s"])
        for row in rows
        if row["policy"] == "replay_only" and row["commit_100_s"]
    ]
    if not rates:
        raise ValueError("width-8 run has no complete replay-only episodes")
    slower, central, faster = np.quantile(rates, (.25, .5, .75))
    return {
        "central": float(central), "faster": float(faster),
        "slower": float(slower),
    }, len(rates)


def measured_kv_caps(width8, crossover, profile):
    quantiles = {"slower": .25, "central": .5, "faster": .75}
    plan_ = json.loads((width8 / "plan.json").read_text())
    scenarios = {row["scenario_id"]: row for row in plan_["scenarios"]}
    episodes = [
        row for row in csv.DictReader(
            (width8 / "policy_episodes.csv").open()
        ) if row["policy"] == "kv_only" and row["commit_100_s"]
    ]
    rates = {}
    for row in episodes:
        scenario = scenarios[row["scenario_id"]]
        size = sum(
            profile.case().kv_transfer.sealed_bytes(session["initial_tokens"])
            for session in scenario["sessions"]
        )
        rates.setdefault(float(scenario["bandwidth_mbps"]), []).append(
            size / float(row["commit_100_s"])
        )
    migrations = list(csv.DictReader((crossover / "migrations.csv").open()))
    summaries = {
        row["scenario_id"]: row
        for row in csv.DictReader((crossover / "scenarios.csv").open())
    }
    serial = {}
    for row in migrations:
        if row["method"] == "kv_transfer":
            serial.setdefault(float(row["bandwidth_mbps"]), []).append(
                float(row["measured_kv_bytes"])
                / float(summaries[row["scenario_id"]]["migration_s"])
            )
    if set(rates) != {5000.0, 10000.0} \
            or not {1000.0, 2500.0} <= set(serial):
        raise ValueError("KV calibration grid is incomplete")
    combined = {1000.0: serial[1000.0], 2500.0: serial[2500.0], **rates}
    return {
        case: {
            bandwidth: float(np.quantile(values, quantile))
            for bandwidth, values in combined.items()
        }
        for case, quantile in quantiles.items()
    }, {"serial": len(serial[1000.0]) + len(serial[2500.0]),
        "width8": sum(map(len, rates.values()))}


def parallel_profile(profile, width, replay_caps):
    if set(replay_caps) != set(profile.cases):
        raise ValueError("replay caps must cover every profile case")
    cases = {}
    for case_id, case in profile.cases.items():
        actions = {}
        for name, curve in case.action_power_w.items():
            actions[name] = ActionPower(
                np.array([1, width]),
                np.array([curve.source_w[0], width * curve.source_w[0]]),
                np.array([curve.destination_w[0],
                          width * curve.destination_w[0]]),
            )
        replay = RateCurve({
            concurrency: (
                case.replay.by_concurrency[1] if concurrency == 1 else (
                    case.replay.by_concurrency[1][0],
                    np.minimum(
                        case.replay.by_concurrency[1][1],
                        replay_caps[case_id] / concurrency,
                    ),
                )
            )
            for concurrency in range(1, width + 1)
        })
        cases[case_id] = replace(
            case, action_power_w=actions, replay=replay
        )
    return replace(
        profile, max_destination_replays=width,
        max_destination_kv_streams=width, cases=cases,
    )


def aggregate_planning_profile(profile, bandwidth_mbps, replay_caps, kv_caps):
    cases = {}
    for case_id, case in profile.cases.items():
        x, y = case.replay.by_concurrency[1]
        replay = RateCurve({1: (x, np.minimum(y, replay_caps[case_id]))})
        transfer = replace(
            case.kv_transfer,
            destination_bytes_per_s=min(
                case.kv_transfer.destination_bytes_per_s,
                kv_caps[case_id][float(bandwidth_mbps)],
            ),
        )
        cases[case_id] = replace(
            case, replay=replay, kv_transfer=transfer
        )
    return replace(
        profile, max_destination_replays=1,
        max_destination_kv_streams=1, cases=cases,
    )


def shared_kv_profile(profile, bandwidth_mbps, concurrency, kv_caps):
    if concurrency <= 1:
        return profile
    cases = {}
    for case_id, case in profile.cases.items():
        cap = kv_caps[case_id][float(bandwidth_mbps)]
        transfer = replace(
            case.kv_transfer,
            destination_bytes_per_s=min(
                case.kv_transfer.destination_bytes_per_s, cap
            ),
        )
        cases[case_id] = replace(case, kv_transfer=transfer)
    return replace(profile, cases=cases)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def admitted_moves(policy, scenario, routes, profile, seed):
    solver = {"queue_haul": "lp_work_first"}.get(policy, policy)
    return tuple(
        PlannedMove(
            move.session_id, move.destination_instance, move.method, move.order,
            move.path,
        )
        for move in plan(scenario, profile, routes, solver, seed=seed).moves
    )


def frontier_metrics(commits, total_sessions, budget_s, power_curve, power_window_s):
    attainment = deadline_attainment(
        commits, total_sessions, [budget_s], power_curve, power_window_s
    )[0]["power_attainment_fraction"]
    return attainment, max(commits, default=0.0)


def workload_grid(fixed_plan, hardware_plan):
    fixed = {}
    for row in fixed_plan["scenarios"]:
        if row["policy"] == "control":
            fixed.setdefault(row["context_profile"], row)
    hardware = {}
    for row in hardware_plan["scenarios"]:
        if row["policy"] == "control":
            hardware.setdefault(row["sample_id"], row)
    bandwidths = sorted({
        row["bandwidth_mbps"] for row in fixed_plan["scenarios"]
        if row["policy"] == "control"
    })
    return [
        (source, row, bandwidth)
        for source, samples in (
            ("fixed_anchor", fixed.values()),
            ("measured_workload_mix", hardware.values()),
        )
        for row in samples
        for bandwidth in bandwidths
    ]


def simulate(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
             crossover_path=DEFAULT_CROSSOVER, width8=DEFAULT_WIDTH8,
             time_budgets_s=TIME_BUDGETS_S):
    plan_ = json.loads(plan_path.read_text())
    validate_policy_plan(plan_)
    hardware_plan = json.loads((width8 / "plan.json").read_text())
    validate_policy_plan(hardware_plan)
    if profiler.file_hash(model_path) != plan_["model_profile"]["sha256"]:
        raise RuntimeError("model profile changed after planning")
    base_profile = ModelProfile.load(model_path)
    replay_caps, _ = measured_replay_caps(width8)
    kv_caps, _ = measured_kv_caps(
        width8, crossover_path.parent, base_profile
    )
    profile = parallel_profile(
        base_profile, plan_["sessions_per_episode"], replay_caps
    )
    crossover = json.loads(crossover_path.read_text())
    anchors = set(crossover["contexts"])
    rows = []
    for configuration, (workload_source, base, bandwidth) in enumerate(
            workload_grid(plan_, hardware_plan)):
        evidence = context_evidence(
            (row["initial_tokens"] for row in base["sessions"]), anchors
        )
        for budget_s in time_budgets_s:
            scenario, routes = _problem(
                profile, base["sessions"], bandwidth, budget_s,
            )
            scenario = replace(scenario, end_s=max(OBSERVATION_S, budget_s))
            planning_profile = aggregate_planning_profile(
                base_profile, bandwidth, replay_caps, kv_caps
            )
            match_id = profiler.object_hash([
                base["sample_id"], bandwidth, budget_s,
            ])[:16]
            for policy in POLICIES:
                moves = admitted_moves(
                    policy, scenario, routes, planning_profile,
                    profiler.stable_seed(
                        plan_["seed"], base["sample_id"],
                        bandwidth, budget_s, policy,
                    ),
                )
                execution_profile = shared_kv_profile(
                    profile, bandwidth,
                    sum(move.method == "kv_transfer" for move in moves), kv_caps,
                )
                result = execute(scenario, execution_profile, moves)
                commits = [row.committed_s for row in result.sessions
                           if row.committed_s is not None]
                if len(commits) != len(moves):
                    raise RuntimeError(
                        f"configuration {configuration} budget {budget_s:g}s "
                        f"{policy} committed {len(commits)}/{len(moves)} admitted"
                    )
                attainment, completion = frontier_metrics(
                    commits, len(scenario.sessions), budget_s,
                    profile.case().power_curve, profile.power_window_s,
                )
                rows.append({
                    "configuration": configuration, "match_id": match_id,
                    "sample_id": base["sample_id"],
                    "context_profile": base["context_profile"],
                    "workload_source": workload_source,
                    "bandwidth_mbps": bandwidth,
                    "time_budget_s": budget_s, "policy": policy,
                    "power_attainment_fraction": attainment,
                    "completion_s": completion,
                    "completion_budget_ratio": completion / budget_s,
                    "full_shed_by_budget": meets_deadline(
                        attainment, completion, budget_s
                    ),
                    "admitted_moves": len(moves),
                    "replay_moves": sum(row.method == "replay"
                                        for row in result.sessions),
                    "kv_moves": sum(row.method == "kv_transfer"
                                    for row in result.sessions),
                    "context_evidence": evidence,
                    "replay_contention_evidence":
                        "measured_width8_aggregate_throughput_cap",
                    "kv_contention_evidence":
                        "measured_bandwidth_specific_aggregate_throughput_cap",
                    "power_evidence": "modeled",
                    "result_evidence": "simulated",
                    "planning_evidence":
                        "deadline_specific_admitted_set_with_aggregate_caps",
                })
    pareto_flags(rows, ("match_id",))
    for row in rows:
        row["paired_pareto"] = row.pop("pareto")
    return rows


def summarize(rows):
    output = []
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        output.append({
            "policy": policy, "scenarios": len(selected),
            "median_power_attainment_fraction": float(np.median([
                row["power_attainment_fraction"] for row in selected
            ])),
            "median_completion_budget_ratio": float(np.median([
                row["completion_budget_ratio"] for row in selected
            ])),
            "deadline_met_fraction": float(np.mean([
                row["full_shed_by_budget"] for row in selected
            ])),
            "paired_pareto_fraction": float(np.mean([
                row["paired_pareto"] for row in selected
            ])),
            "median_admitted_moves": float(np.median([
                row["admitted_moves"] for row in selected
            ])),
        })
    return output


def full_attainment_cdf(rows, policy, threshold=.99):
    values = sorted(
        row["completion_budget_ratio"] for row in rows
        if row["policy"] == policy
        and row["power_attainment_fraction"] >= threshold
    )
    return np.asarray(values), np.arange(1, len(values) + 1) / len(values) \
        if values else np.array([])


def plot(rows, out):
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
    markers = dict(zip(POLICIES, ("o", "s", "^", "D", "x")))
    budgets = sorted({row["time_budget_s"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cloud, mixture = axes
    for policy in (*POLICIES[1:], POLICIES[0]):
        selected = [row for row in rows if row["policy"] == policy]
        style = {
            "s": 26, "alpha": .6, "facecolors": "none",
            "edgecolors": colors[policy], "linewidths": 1, "zorder": 4,
        } if policy == "queue_haul" else {
            "s": 18, "alpha": .3, "color": colors[policy],
        }
        cloud.scatter(
            [100 * row["power_attainment_fraction"] for row in selected],
            [row["completion_budget_ratio"] for row in selected],
            marker=markers[policy], label=LABELS[policy], **style,
        )
        aggregate = [(
            budget,
            np.mean([
                row["power_attainment_fraction"] for row in selected
                if row["time_budget_s"] == budget
            ]),
            np.median([
                row["completion_s"] for row in selected
                if row["time_budget_s"] == budget
            ]),
        ) for budget in budgets]
        mixture.plot(
            [100 * row[1] for row in aggregate],
            [row[2] for row in aggregate],
            marker=markers[policy], color=colors[policy], label=LABELS[policy],
            linewidth=2.5 if policy == "queue_haul" else 1.5,
            zorder=3 if policy == "queue_haul" else 2,
        )
    cloud.axhline(1, color="0.35", linestyle=":", linewidth=1)
    cloud.set(
        title="Matched scenario–budget outcomes",
        xlabel="Modeled maximum source-power shed by deadline (%)",
        ylabel="Admitted-set completion / time budget",
        xlim=(-2, 102),
    )
    cloud.grid(alpha=.2)
    mixture.set(
        title="Identical workload mixture at each time budget",
        xlabel="Mean modeled source-power shed (%)",
        ylabel="Median admitted-set completion (s)",
        xlim=(-2, 102),
    )
    mixture.grid(alpha=.2)
    handles, labels = cloud.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    cloud.text(
        .01, .01,
        "2/4/8/16/24/32K replay anchors; fixed anchors + measured workload mixes\n"
        "non-anchor rates interpolated; action power extrapolated; power modeled",
        transform=cloud.transAxes, fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0, 1, .82))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"simulated_width8_pareto.{suffix}", dpi=220)
    plt.close(fig)


def run(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
        crossover_path=DEFAULT_CROSSOVER, width8=DEFAULT_WIDTH8,
        out=DEFAULT_OUT):
    out.mkdir(parents=True, exist_ok=True)
    rows = simulate(plan_path, model_path, crossover_path, width8)
    summary = summarize(rows)
    replay_caps, replay_episodes = measured_replay_caps(width8)
    profile = ModelProfile.load(model_path)
    kv_caps, kv_episodes = measured_kv_caps(
        width8, crossover_path.parent, profile
    )
    write_csv(out / "simulated_pareto.csv", rows)
    write_csv(out / "policy_summary.csv", summary)
    plot(rows, out)
    metadata = {
        "schema": "queue-haul-simulated-pareto-v2",
        "axes": {
            "x": "deadline-integrated source-power shed / removable power",
            "y": "last admitted route commit time",
            "mixture": "mean shed and median completion over identical workloads",
        },
        "policies": list(POLICIES),
        "scenarios": len(rows),
        "paired_episodes": len(rows) // len(POLICIES),
        "time_budgets_s": list(TIME_BUDGETS_S),
        "observation_s": OBSERVATION_S,
        "planning_contract":
            "deadline-specific admitted set; no appended cleanup moves; eager execution",
        "workload_contract":
            "five fixed anchors plus 18 measured width-8 workload mixes, crossed "
            "with every bandwidth and time budget",
        "plan": {
            "path": _portable_path(plan_path),
            "sha256": profiler.file_hash(plan_path),
        },
        "model": {
            "path": _portable_path(model_path),
            "sha256": profiler.file_hash(model_path),
        },
        "crossover": {
            "path": _portable_path(crossover_path),
            "sha256": profiler.file_hash(crossover_path),
        },
        "width8_replay_cap": {
            "plan": {
                "path": _portable_path(width8 / "plan.json"),
                "sha256": profiler.file_hash(width8 / "plan.json"),
            },
            "episodes": {
                "path": _portable_path(width8 / "policy_episodes.csv"),
                "sha256": profiler.file_hash(width8 / "policy_episodes.csv"),
            },
            "complete_replay_only_episodes": replay_episodes,
            "aggregate_tokens_per_s": replay_caps,
        },
        "kv_aggregate_caps": {
            "crossover_migrations": kv_episodes["serial"],
            "width8_episodes": kv_episodes["width8"],
            "aggregate_bytes_per_s_by_case_and_bandwidth": kv_caps,
        },
        "evidence": {
            "context_anchors": "measured",
            "in_range_nonanchors": "interpolated",
            "width8_launch": "measured in policy-hardware-width8-frontier-20260730",
            "width8_replay_aggregate_rate":
                "measured replay-only episode context / completion time",
            "kv_aggregate_rate":
                "serial at 1/2.5 Gbit/s; width-8 at 5/10 Gbit/s",
            "width8_action_power": "extrapolated from serial calibration",
            "power_attainment": "modeled from commit times",
            "results": "simulated",
        },
        "paired_pareto": "dominance is evaluated only within each matched episode",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = sorted(path for path in out.iterdir() if path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in files
    ))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--crossover-plan", type=Path, default=DEFAULT_CROSSOVER)
    parser.add_argument("--width8-run", type=Path, default=DEFAULT_WIDTH8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(
        args.plan, args.model_profile, args.crossover_plan,
        args.width8_run, args.out,
    )


if __name__ == "__main__":
    main()
