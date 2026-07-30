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
    LABELS, _moves, _portable_path, _problem, deadline_attainment,
    validate_policy_plan,
)
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
            and other["completion_deadline_ratio"]
            <= row["completion_deadline_ratio"]
            and (
                other["power_attainment_fraction"]
                > row["power_attainment_fraction"]
                or other["completion_deadline_ratio"]
                < row["completion_deadline_ratio"]
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


def planned_moves(rows):
    return tuple(
        PlannedMove(
            row["session_id"], "destination", row["method"], row["order"],
            ("link",),
        )
        for row in sorted(rows, key=lambda row: row["order"])
    )


def simulate(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
             crossover_path=DEFAULT_CROSSOVER, width8=DEFAULT_WIDTH8):
    plan_ = json.loads(plan_path.read_text())
    validate_policy_plan(plan_)
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
    by_episode = {}
    for scenario in plan_["scenarios"]:
        by_episode.setdefault(scenario["episode"], {})[scenario["policy"]] = scenario
    rows = []
    for episode, scenarios in sorted(by_episode.items()):
        base = scenarios["control"]
        scenario, routes = _problem(
            profile, base["sessions"], base["bandwidth_mbps"],
            base["required_deadline_s"],
        )
        scenario = replace(scenario, end_s=max(base["deadline_s"], OBSERVATION_S))
        planning_profile = aggregate_planning_profile(
            base_profile, base["bandwidth_mbps"], replay_caps, kv_caps
        )
        evidence = context_evidence(
            (row["initial_tokens"] for row in base["sessions"]), anchors
        )
        for policy in POLICIES:
            moves = _moves(
                policy, scenario, routes, planning_profile,
                profiler.stable_seed(plan_["seed"], episode, policy),
            )
            moves = planned_moves(moves)
            execution_profile = shared_kv_profile(
                profile, base["bandwidth_mbps"],
                sum(move.method == "kv_transfer" for move in moves), kv_caps,
            )
            result = execute(scenario, execution_profile, moves)
            commits = [row.committed_s for row in result.sessions
                       if row.committed_s is not None]
            if len(commits) != len(scenario.sessions):
                raise RuntimeError(
                    f"episode {episode} {policy} committed "
                    f"{len(commits)}/{len(scenario.sessions)} by {scenario.end_s}s"
                )
            completion = max(commits)
            attainment = deadline_attainment(
                commits, len(scenario.sessions),
                [base["required_deadline_s"]], profile.case().power_curve,
                profile.power_window_s,
            )[0]["power_attainment_fraction"]
            rows.append({
                "episode": episode, "match_id": base["match_id"],
                "context_profile": base["context_profile"],
                "bandwidth_mbps": base["bandwidth_mbps"],
                "required_deadline_s": base["required_deadline_s"],
                "repeat": base["repeat"], "policy": policy,
                "power_attainment_fraction": attainment,
                "completion_s": completion,
                "completion_deadline_ratio":
                    completion / base["required_deadline_s"],
                "deadline_met": meets_deadline(
                    attainment, completion, base["required_deadline_s"]
                ),
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
                "planning_evidence": "replanned_with_aggregate_caps",
            })
    pareto_flags(rows, ("match_id",))
    for row in rows:
        row["paired_pareto"] = row.pop("pareto")
    pareto_flags(rows, ())
    for row in rows:
        row["pooled_pareto"] = row.pop("pareto")
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
            "median_completion_deadline_ratio": float(np.median([
                row["completion_deadline_ratio"] for row in selected
            ])),
            "deadline_met_fraction": float(np.mean([
                row["deadline_met"] for row in selected
            ])),
            "paired_pareto_fraction": float(np.mean([
                row["paired_pareto"] for row in selected
            ])),
            "pooled_pareto_points": sum(
                row["pooled_pareto"] for row in selected
            ),
        })
    return output


def full_attainment_cdf(rows, policy, threshold=.99):
    values = sorted(
        row["completion_deadline_ratio"] for row in rows
        if row["policy"] == policy
        and row["power_attainment_fraction"] >= threshold
    )
    return np.asarray(values), np.arange(1, len(values) + 1) / len(values) \
        if values else np.array([])


def plot(rows, out):
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
    markers = dict(zip(POLICIES, ("o", "s", "^", "D", "x")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax, detail = axes
    for policy in (*POLICIES[1:], POLICIES[0]):
        selected = [row for row in rows if row["policy"] == policy]
        style = {
            "s": 52, "alpha": .9, "facecolors": "none",
            "edgecolors": colors[policy], "linewidths": 1.5, "zorder": 4,
        } if policy == "queue_haul" else {
            "s": 28, "alpha": .45, "color": colors[policy],
        }
        ax.scatter(
            [100 * row["power_attainment_fraction"] for row in selected],
            [row["completion_deadline_ratio"] for row in selected],
            marker=markers[policy], label=LABELS[policy], **style,
        )
        x, y = full_attainment_cdf(rows, policy)
        detail.step(
            np.r_[0, x], np.r_[0, y], where="post", color=colors[policy],
            linewidth=2.5 if policy == "queue_haul" else 1.5,
            zorder=3 if policy == "queue_haul" else 2,
        )
    frontier = sorted(
        (row for row in rows if row["pooled_pareto"]),
        key=lambda row: row["power_attainment_fraction"],
    )
    ax.plot(
        [100 * row["power_attainment_fraction"] for row in frontier],
        [row["completion_deadline_ratio"] for row in frontier],
        "k--", linewidth=1.5, label="Pooled descriptive frontier",
    )
    ax.axhline(1, color="0.35", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Modeled maximum source-power shed by deadline (%)",
        ylabel="Completion time / required deadline",
        xlim=(-2, 102),
    )
    ax.grid(alpha=.2)
    detail.axvline(1, color="0.35", linestyle=":", linewidth=1)
    detail.set(
        title="Detail: scenarios attaining ≥99% shed",
        xlabel="Completion time / required deadline",
        ylabel="Fraction of full-attainment scenarios",
        xlim=(0, 1.02), ylim=(0, 1.02),
    )
    detail.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    ax.text(
        .01, .01,
        "2/4/8/16K contexts measured; 12/14K interpolated\n"
        "replay/KV aggregate caps measured; action power extrapolated; power modeled",
        transform=ax.transAxes, fontsize=8, va="bottom",
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
        "schema": "queue-haul-simulated-pareto-v1",
        "axes": {
            "x": "commit-integrated source-power shed / removable power",
            "y": "last route commit time / required deadline",
            "detail": "per-policy completion CDF for attainment >= 0.99",
        },
        "policies": list(POLICIES),
        "scenarios": len(rows),
        "paired_episodes": len(rows) // len(POLICIES),
        "observation_s": OBSERVATION_S,
        "planning_contract":
            "replanned shared aggregate replay/KV capacity; eager execution",
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
        "pooled_frontier": "descriptive across heterogeneous scenarios",
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
