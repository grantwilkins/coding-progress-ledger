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


def parallel_profile(profile, width):
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
            concurrency: case.replay.by_concurrency[1]
            for concurrency in range(1, width + 1)
        })
        cases[case_id] = replace(
            case, action_power_w=actions, replay=replay
        )
    return replace(
        profile, max_destination_replays=width,
        max_destination_kv_streams=width, cases=cases,
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def planned_moves(rows):
    return tuple(
        PlannedMove(
            row["session_id"], "destination", row["method"], row["order"],
            ("link",), row.get("planned_rate_limit_bytes_per_s"),
            row.get("planned_quiesce_s"),
        )
        for row in sorted(rows, key=lambda row: row["order"])
    )


def simulate(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
             crossover_path=DEFAULT_CROSSOVER):
    plan_ = json.loads(plan_path.read_text())
    validate_policy_plan(plan_)
    if profiler.file_hash(model_path) != plan_["model_profile"]["sha256"]:
        raise RuntimeError("model profile changed after planning")
    profile = ModelProfile.load(model_path)
    profile = parallel_profile(profile, plan_["sessions_per_episode"])
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
        evidence = context_evidence(
            (row["initial_tokens"] for row in base["sessions"]), anchors
        )
        for policy in POLICIES:
            moves = scenarios.get(policy, {}).get("moves")
            if moves is None:
                moves = _moves(
                    policy, scenario, routes, profile,
                    profiler.stable_seed(plan_["seed"], episode, policy),
                )
            result = execute(scenario, profile, planned_moves(moves))
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
                "contention_evidence":
                    "measured_parallel_launch_extrapolated_per_stream_rate",
                "power_evidence": "modeled",
                "result_evidence": "simulated",
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


def plot(rows, out):
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
    markers = dict(zip(POLICIES, ("o", "s", "^", "D", "x")))
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        ax.scatter(
            [100 * row["power_attainment_fraction"] for row in selected],
            [row["completion_deadline_ratio"] for row in selected],
            s=28, alpha=.55, color=colors[policy], marker=markers[policy],
            label=LABELS[policy],
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
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    ax.text(
        .01, .01,
        "2/4/8/16K contexts measured; 12/14K interpolated\n"
        "width-8 launch measured; per-stream rates extrapolated; power modeled",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0, 1, .86))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"simulated_width8_pareto.{suffix}", dpi=220)
    plt.close(fig)


def run(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
        crossover_path=DEFAULT_CROSSOVER, out=DEFAULT_OUT):
    out.mkdir(parents=True, exist_ok=True)
    rows = simulate(plan_path, model_path, crossover_path)
    summary = summarize(rows)
    write_csv(out / "simulated_pareto.csv", rows)
    write_csv(out / "policy_summary.csv", summary)
    plot(rows, out)
    metadata = {
        "schema": "queue-haul-simulated-pareto-v1",
        "axes": {
            "x": "commit-integrated source-power shed / removable power",
            "y": "last route commit time / required deadline",
        },
        "policies": list(POLICIES),
        "scenarios": len(rows),
        "paired_episodes": len(rows) // len(POLICIES),
        "observation_s": OBSERVATION_S,
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
        "evidence": {
            "context_anchors": "measured",
            "in_range_nonanchors": "interpolated",
            "width8_launch": "measured in policy-hardware-width8-frontier-20260730",
            "width8_per_stream_rate_and_action_power":
                "extrapolated from serial calibration",
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.plan, args.model_profile, args.crossover_plan, args.out)


if __name__ == "__main__":
    main()
