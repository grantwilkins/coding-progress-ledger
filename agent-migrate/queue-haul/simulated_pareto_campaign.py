"""Simulate full-width fluid migration plans and plot their frontier."""

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
from destination import (
    DESTINATION_SCHEMA, CompatibilityFingerprint, ContextRate,
    DestinationArchitecture, DestinationPool, DestinationReplica,
    DestinationType, LoadedCoefficients,
)
from policy_hardware_campaign import (
    LABELS, _portable_path, _problem, deadline_attainment,
    validate_policy_plan,
)
from planner import plan
from profiles import ActionPower, ModelProfile, RateCurve
from simulate import predict


ROOT = Path(__file__).parent
DEFAULT_PLAN = ROOT / "outputs/policy-hardware-width8-packing-plan/plan.json"
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
DEFAULT_CROSSOVER = ROOT / "outputs/policy-hardware-crossover-20260730/plan.json"
DEFAULT_WORKLOAD_PLAN = ROOT / "outputs/policy-hardware-width8-frontier-20260730/plan.json"
DEFAULT_OUT = ROOT / "outputs/simulated-fullwidth-pareto-20260802"
DEFAULT_SESSIONS = 10_000
MAX_FLUID_PLANNING_SESSIONS = 256
POLICIES = (
    "queue_haul", "greedy", "greedy_coupled", "random", "kv_only",
    "replay_only",
)
LABELS = {**LABELS, "greedy_coupled": "Coupled greedy"}
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


def fluid_profile(profile, width, resident_tokens):
    if width < 1 or resident_tokens < 1:
        raise ValueError("fluid profile dimensions must be positive")
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
        replay = RateCurve({concurrency: case.replay.by_concurrency[1]
                            for concurrency in range(1, width + 1)})
        cases[case_id] = replace(
            case, action_power_w=actions, replay=replay
        )
    return replace(
        profile, kv_capacity_tokens=max(profile.kv_capacity_tokens, resident_tokens),
        max_destination_replays=width, max_destination_kv_streams=width,
        cases=cases,
    )


def aggregate_profile(profile, base_profile, width, replication, resident_tokens):
    if replication < 1:
        raise ValueError("replication must be positive")
    grouped = fluid_profile(base_profile, width, resident_tokens)
    cases = {}
    for case_id, case in grouped.cases.items():
        actions = {
            name: replace(
                curve,
                source_w=replication * curve.source_w,
                destination_w=replication * curve.destination_w,
            )
            for name, curve in case.action_power_w.items()
        }
        kv_transfer = replace(
            case.kv_transfer,
            destination_bytes_per_s=case.kv_transfer.destination_bytes_per_s
            / replication,
        )
        cases[case_id] = replace(
            case, action_power_w=actions, kv_transfer=kv_transfer,
        )
    return replace(
        grouped,
        kv_capacity_tokens=max(1, profile.kv_capacity_tokens // replication),
        cases=cases,
    )


def expand_sessions(base, sessions):
    if sessions < 1:
        raise ValueError("sessions must be positive")
    templates = base["sessions"]
    return [{
        "session_id": f"{base['sample_id']}-{index}",
        "job_class": templates[index % len(templates)]["job_class"],
        "turn_index": 0,
        "initial_tokens": templates[index % len(templates)]["initial_tokens"],
        "order": index,
    } for index in range(sessions)]


def expand_moves(moves, template_rows, session_rows, template_width=None):
    by_template = {move.session_id: move for move in moves}
    replication = len(session_rows) // len(template_rows)
    template_width = template_width or len(template_rows)
    expanded = []
    for row in session_rows:
        template = template_rows[
            (row["order"] // template_width // replication) * template_width
            + row["order"] % template_width
        ]
        if template["session_id"] in by_template:
            expanded.append(replace(
                by_template[template["session_id"]],
                session_id=row["session_id"], order=len(expanded),
            ))
    return tuple(expanded)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coupled_architecture(profile):
    case = profile.case()
    fingerprint = CompatibilityFingerprint(
        profile.model, "gpt-oss-pinned", "source-dc-log", "lmcache-mp-v7",
    )
    def rate(curve):
        return ContextRate(*(
            tuple(map(float, values))
            for values in curve.by_concurrency[1]
        ))
    contexts = tuple(map(float, case.prefill.by_concurrency[1][0]))
    loaded = LoadedCoefficients(
        (0, 1), (1, 1), (contexts[0], contexts[-1]),
        (125_000_000, 1_250_000_000),
        "simulated-pareto-zero-load-sensitivity",
    )
    destination_type = DestinationType(
        "gpt-oss-20b-a100-tp1", fingerprint, rate(case.prefill),
        rate(case.decode), ((1, 1),),
        {mode: (1,) for mode in ("normal", "emergency", "stable")},
        profile.kv_capacity_tokens,
        {"replay": loaded, "kv_transfer": loaded}, (0, 1),
        "simulated-pareto-zero-load-sensitivity", True,
        case.kv_transfer.block_tokens,
    )
    pool = DestinationPool(
        "coupled-pool", destination_type.type_id,
        (DestinationReplica("destination"),), "source-to-destination", ("link",),
    )
    return DestinationArchitecture(
        DESTINATION_SCHEMA, fingerprint, (destination_type,), (pool,),
    )


def admitted_moves(policy, scenario, routes, profile, seed, destination=None,
                   replication=1):
    solver = {"queue_haul": "lp_work_first"}.get(policy, policy)
    return tuple(
        replace(
            move, rate_limit_bytes_per_s=None, quiesce_s=None,
        )
        for move in plan(
            scenario, profile, routes, solver, seed=seed,
            destination=destination, replication=replication,
        ).moves
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
             crossover_path=DEFAULT_CROSSOVER,
             workload_plan=DEFAULT_WORKLOAD_PLAN, sessions=DEFAULT_SESSIONS,
             time_budgets_s=TIME_BUDGETS_S):
    plan_ = json.loads(plan_path.read_text())
    validate_policy_plan(plan_)
    workload_plan_ = json.loads(workload_plan.read_text())
    validate_policy_plan(workload_plan_)
    if sessions < 1:
        raise ValueError("sessions must be positive")
    if profiler.file_hash(model_path) != plan_["model_profile"]["sha256"]:
        raise RuntimeError("model profile changed after planning")
    base_profile = ModelProfile.load(model_path)
    crossover = json.loads(crossover_path.read_text())
    anchors = set(crossover["contexts"])
    rows = []
    for configuration, (workload_source, base, bandwidth) in enumerate(
            workload_grid(plan_, workload_plan_)):
        template_width = len(base["sessions"])
        if not template_width or sessions % template_width:
            raise ValueError(
                "sessions must be a positive multiple of the workload template width"
            )
        replications = sessions // template_width
        planning_repeat = min(
            replications, MAX_FLUID_PLANNING_SESSIONS // template_width,
        )
        while replications % planning_repeat:
            planning_repeat -= 1
        planning_width = template_width * planning_repeat
        group_replications = replications // planning_repeat
        template_rows = expand_sessions(base, planning_width)
        session_rows = expand_sessions(base, sessions)
        profile = fluid_profile(
            base_profile, sessions,
            sum(row["initial_tokens"] for row in session_rows),
        )
        planning_profile = replace(
            profile,
            cases=fluid_profile(
                base_profile, planning_width,
                sum(row["initial_tokens"] for row in template_rows),
            ).cases,
        )
        execution_profile = aggregate_profile(
            profile, base_profile, planning_width, group_replications,
            sum(row["initial_tokens"] for row in template_rows),
        )
        evidence = context_evidence(
            (row["initial_tokens"] for row in session_rows), anchors
        )
        for budget_s in time_budgets_s:
            scenario, routes = _problem(
                execution_profile, template_rows, bandwidth / group_replications,
                budget_s,
            )
            scenario = replace(
                scenario,
                end_s=max(OBSERVATION_S, budget_s) * replications,
            )
            planning_scenario, planning_routes = _problem(
                planning_profile, template_rows, bandwidth,
                budget_s,
            )
            planning_scenario = replace(
                planning_scenario, end_s=max(OBSERVATION_S, budget_s)
            )
            match_id = profiler.object_hash([
                base["sample_id"], sessions, bandwidth, budget_s,
            ])[:16]
            for policy in POLICIES:
                destination = coupled_architecture(execution_profile) \
                    if policy == "greedy_coupled" else None
                template_moves = admitted_moves(
                    policy, planning_scenario, planning_routes, planning_profile,
                    profiler.stable_seed(
                        plan_["seed"], base["sample_id"],
                        bandwidth, budget_s, policy,
                    ),
                    destination,
                    group_replications,
                )
                moves = expand_moves(
                    template_moves, template_rows, session_rows, template_width,
                )
                result = predict(
                    scenario, execution_profile, template_moves,
                    destination=destination,
                )
                commits = [row.committed_s
                           for row in result.sessions
                           if row.committed_s is not None
                           for _ in range(group_replications)]
                if len(commits) != len(moves):
                    raise RuntimeError(
                        f"configuration {configuration} budget {budget_s:g}s "
                        f"{policy} committed {len(commits)}/{len(moves)} admitted"
                    )
                attainment, completion = frontier_metrics(
                    commits, sessions, budget_s,
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
                    "replay_moves": sum(move.method == "replay" for move in moves),
                    "kv_moves": sum(move.method == "kv_transfer" for move in moves),
                    "context_evidence": evidence,
                    "replay_contention_evidence":
                        "serial_measured_rate_per_active_flow",
                    "kv_contention_evidence":
                        "shared_destination_and_route_links",
                    "power_evidence": "modeled",
                    "result_evidence": "simulated",
                    "planning_evidence":
                        "deadline_specific_admitted_set_with_fluid_links",
                    "destination_contract":
                        "single_pool_zero_load_service_sensitivity"
                        if destination else "fluid_shared_link_execution",
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


def policy_coordinates(rows, policy, normalized):
    selected = [row for row in rows if row["policy"] == policy]
    return (
        [100 * row["power_attainment_fraction"] for row in selected],
        [row["completion_budget_ratio" if normalized else "completion_s"]
         for row in selected],
    )


def plot(rows, out):
    colors = dict(zip(POLICIES, plt.get_cmap("tab10").colors))
    markers = dict(zip(POLICIES, ("o", "s", "P", "^", "D", "x")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cloud, raw = axes
    for policy in (*POLICIES[1:], POLICIES[0]):
        style = {
            "s": 26, "alpha": .6, "facecolors": "none",
            "edgecolors": colors[policy], "linewidths": 1, "zorder": 4,
        } if policy == "queue_haul" else {
            "s": 18, "alpha": .3, "color": colors[policy],
        }
        for axis, normalized in ((cloud, True), (raw, False)):
            axis.scatter(
                *policy_coordinates(rows, policy, normalized),
                marker=markers[policy], label=LABELS[policy], **style,
            )
    cloud.axhline(1, color="0.35", linestyle=":", linewidth=1)
    cloud.set(
        title="Matched scenario–budget outcomes",
        xlabel="Modeled maximum source-power shed by deadline (%)",
        ylabel="Admitted-set completion / time budget",
        xlim=(-2, 102),
    )
    cloud.grid(alpha=.2)
    raw.set(
        title="All admitted-set completions",
        xlabel="Modeled maximum source-power shed by deadline (%)",
        ylabel="Admitted-set completion (s)",
        xlim=(-2, 102),
    )
    raw.grid(alpha=.2)
    handles, labels = cloud.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=3)
    cloud.text(
        .01, .01,
        "Full-width fluid episodes; fixed anchors + measured workload mixes\n"
        "non-anchor rates interpolated; serial power extended; power modeled",
        transform=cloud.transAxes, fontsize=8, va="bottom",
    )
    fig.tight_layout(rect=(0, 0, 1, .82))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"simulated_fullwidth_pareto.{suffix}", dpi=220)
    plt.close(fig)


def run(plan_path=DEFAULT_PLAN, model_path=DEFAULT_MODEL,
        crossover_path=DEFAULT_CROSSOVER,
        workload_plan=DEFAULT_WORKLOAD_PLAN, sessions=DEFAULT_SESSIONS,
        out=DEFAULT_OUT):
    out.mkdir(parents=True, exist_ok=True)
    rows = simulate(
        plan_path, model_path, crossover_path, workload_plan, sessions,
    )
    summary = summarize(rows)
    write_csv(out / "simulated_pareto.csv", rows)
    write_csv(out / "policy_summary.csv", summary)
    plot(rows, out)
    metadata = {
        "schema": "queue-haul-simulated-pareto-v3",
        "axes": {
            "x": "deadline-integrated source-power shed / removable power",
            "y": "last admitted route commit time",
            "raw_panel": "one point per matched scenario-budget-policy result",
        },
        "policies": list(POLICIES),
        "scenarios": len(rows),
        "paired_episodes": len(rows) // len(POLICIES),
        "time_budgets_s": list(TIME_BUDGETS_S),
        "observation_s": OBSERVATION_S,
        "planning_contract":
            "deadline-specific admitted template set; replicated groups share fluid resources; "
            "no appended cleanup moves; eager execution",
        "execution_contract":
            "each template group represents identical full-width flows; route bandwidth and "
            "destination ingest are divided by replication while action power is multiplied",
        "greedy_coupled_contract":
            "one destination replica and measured link; zero background load and "
            "destination service headroom are sensitivity inputs",
        "workload_contract":
            "five fixed anchors plus measured workload templates, expanded to the "
            "requested episode width and crossed with every bandwidth and budget",
        "sessions_per_episode": sessions,
        "fluid_planning_max_sessions": MAX_FLUID_PLANNING_SESSIONS,
        "completion_horizon_contract":
            "max(observation_s, budget_s) scaled by full template replication",
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
        "workload_plan": {
            "path": _portable_path(workload_plan),
            "sha256": profiler.file_hash(workload_plan),
        },
        "evidence": {
            "context_anchors": "measured",
            "in_range_nonanchors": "interpolated",
            "full_width_execution": "fluid shared-link simulation",
            "replay_rate": "serial measured rate applied per active flow",
            "kv_rate": "destination ingest and route links share capacity",
            "action_power": "linear extension of serial calibration",
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
    parser.add_argument("--workload-plan", type=Path, default=DEFAULT_WORKLOAD_PLAN)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(
        args.plan, args.model_profile, args.crossover_plan,
        args.workload_plan, args.sessions, args.out,
    )


if __name__ == "__main__":
    main()
