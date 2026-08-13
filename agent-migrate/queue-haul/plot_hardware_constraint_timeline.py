"""Plot estimated constraint consumption in measured migration order."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import network_campaign as campaign
from pool_planner import candidate_table
from power_model import ExpectedPower


POLICIES = (
    "queue_haul_robust",
    "queue_haul_power_blind",
    campaign.DEADLINE_BLIND_POLICY,
)
RESOURCES = (
    "kv:pool/east",
    "service:pool/germany:0",
    "migration:pool/east:replay",
    "migration:pool/germany:replay",
    "migration:pool/germany:kv_transfer",
)


def cumulative_resource_timeline(completions, work, capacities):
    """Normalize physical work and accumulate it in completion order."""
    if set(completions) != set(work) or any(value <= 0 for value in capacities.values()):
        raise ValueError("timeline inputs do not describe the same positive-capacity moves")
    if any(time_s < 0 for time_s in completions.values()) or any(
            value < 0 for values in work.values() for value in values.values()):
        raise ValueError("timeline time and work must be nonnegative")
    used = dict.fromkeys(capacities, 0.0)
    rows = [(0.0, dict(used))]
    for session_id, time_s in sorted(completions.items(), key=lambda row: (row[1], row[0])):
        for resource in capacities:
            used[resource] += work[session_id].get(resource, 0) / capacities[resource]
        rows.append((time_s, dict(used)))
    return rows


def _resolve(reference: str, plan_path: Path) -> Path:
    path = Path(reference)
    candidates = dict.fromkeys((path, plan_path.parent / path, campaign.ROOT.parent / path))
    resolved = [candidate for candidate in candidates if candidate.is_file()]
    if len(resolved) != 1:
        raise FileNotFoundError(f"cannot resolve exactly one {reference!r}")
    return resolved[0]


def _evidence(raw_root: Path, condition: str, repeat: int) -> dict:
    selected = {}
    for scenario_path in raw_root.glob("scenarios/*/attempt-*/scenario.json"):
        scenario = json.loads(scenario_path.read_text())
        if scenario.get("condition_id") != condition or scenario.get("repeat") != repeat \
                or scenario.get("policy") not in POLICIES:
            continue
        result = json.loads((scenario_path.parent / "result.json").read_text())
        if result.get("status") != "complete":
            continue
        policy = scenario["policy"]
        if policy in selected:
            raise RuntimeError(f"multiple successful attempts for {policy}")
        selected[policy] = (
            scenario,
            json.loads((scenario_path.parent / "decision.json").read_text()),
            result,
        )
    if set(selected) != set(POLICIES):
        raise RuntimeError("missing matched successful policy evidence")
    return selected


def _table(scenario, manifest, profile, blind=False):
    problem, architecture, _routes, _target, _demand = campaign._scenario_problem(
        scenario, manifest, profile)
    if blind:
        problem = replace(problem, deadline_s=campaign.ORACLE_STALE_HORIZON_S,
                          end_s=campaign.ORACLE_STALE_HORIZON_S)
    table = candidate_table(
        problem, profile, architecture, "normal",
        ExpectedPower(replace(problem, final_state="awake", assumed_shutdown_s=None), profile),
    )
    return table, architecture


def estimate(raw_root: Path, plan_path: Path, condition="all-bind", repeat=0):
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(_resolve(plan["manifest"]["path"], plan_path).read_text())
    profile = campaign.ModelProfile.load(campaign.MODEL_PATH)
    evidence = _evidence(raw_root, condition, repeat)
    aware, _ = _table(evidence[POLICIES[0]][0], manifest, profile)
    capacity = dict(zip(aware.resource_names, aware.resource_capacities))
    if any(resource not in capacity for resource in RESOURCES):
        raise RuntimeError("aware plan lacks a plotted resource")
    capacity = {resource: capacity[resource] for resource in RESOURCES}
    timelines, power = {}, {}
    for policy, (scenario, decision, result) in evidence.items():
        table, architecture = _table(
            scenario, manifest, profile, policy == campaign.DEADLINE_BLIND_POLICY)
        candidates = {}
        for column, candidate in enumerate(table.candidates):
            key = (table.sessions[candidate.session].session_id, candidate.method,
                   architecture.pools[candidate.pool].pool_id)
            if key in candidates:
                raise RuntimeError(f"duplicate candidate {key}")
            candidates[key] = column
        work = {}
        for move in decision["moves"]:
            key = (move["session_id"], move["method"], move["destination_pool"])
            if key not in candidates:
                raise RuntimeError(f"missing planned candidate {key}")
            column = candidates[key]
            normalized = table.resources[:, column].toarray().ravel()
            physical = dict(zip(
                table.resource_names,
                normalized * table.resource_capacities,
            ))
            if move["session_id"] in work:
                raise RuntimeError(f"duplicate move for {move['session_id']}")
            work[move["session_id"]] = {
                resource: physical.get(resource, 0.0) for resource in RESOURCES}
        started_ns = int(result["started_ns"])
        completed = [row for row in result["requests"] if "request" in row]
        completions = {row["session_id"]:
                       (int(row["request"]["end_ns"]) - started_ns) / 1e9
                       for row in completed}
        if len(completions) != len(completed):
            raise RuntimeError("duplicate completed session")
        timelines[policy] = cumulative_resource_timeline(completions, work, capacity)
        power[policy] = result["attainment_curve"]
    return timelines, power, evidence[POLICIES[0]][2]["requested_shed_w"]


def write_plot(timelines, power, target, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        POLICIES[0]: "Queue-Haul",
        POLICIES[1]: "Power-blind",
        POLICIES[2]: "Deadline-blind",
    }
    titles = {
        RESOURCES[0]: "East KV headroom",
        RESOURCES[1]: "Germany service headroom",
        RESOURCES[2]: "East replay window",
        RESOURCES[3]: "Germany replay window",
        RESOURCES[4]: "Germany KV-transfer window",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.ravel()
    for policy in POLICIES:
        color = campaign.HARDWARE_GAP_COLORS[policy]
        axes[0].step(
            [row["time_s"] for row in power[policy]],
            [row["shed_w"] for row in power[policy]],
            where="post", color=color, label=labels[policy],
        )
        for axis, resource in zip(axes[1:], RESOURCES):
            axis.step(
                [time_s for time_s, _ in timelines[policy]],
                [values[resource] for _, values in timelines[policy]],
                where="post", color=color,
            )
    axes[0].axhline(target, color="black", linestyle="--", label="Power target")
    axes[0].set(title="Measured source-power attainment", ylabel="Power shed (W)")
    for axis, resource in zip(axes[1:], RESOURCES):
        axis.axhline(1, color="black", linestyle="--")
        axis.set(title=titles[resource], ylabel="30 s budget used")
        axis.set_ylim(0, max(1.08, axis.get_ylim()[1]))
    for index, axis in enumerate(axes):
        axis.axvline(30, color="black", linestyle=":", linewidth=1,
                     label="30 s planner cutoff" if index == 0 else None)
        axis.axvline(45, color="black", linestyle="-.", linewidth=1,
                     label="45 s hardware deadline" if index == 0 else None)
        axis.set_xlim(0, 60)
        axis.set_xlabel("Measured migration time (s)")
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def write_csv(timelines, power, target, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in POLICIES:
        rows.extend({"policy": policy, "metric": "power_shed_w",
                     "time_s": row["time_s"], "value": row["shed_w"],
                     "limit": target, "basis": "measured"}
                    for row in power[policy])
        rows.extend({"policy": policy, "metric": resource, "time_s": time_s,
                     "value": values[resource], "limit": 1,
                     "basis": "planned work in measured completion order"}
                    for time_s, values in timelines[policy] for resource in RESOURCES)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default="all-bind")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    timelines, power, target = estimate(
        args.raw_root, args.plan, args.condition, args.repeat)
    write_csv(timelines, power, target, args.out)
    write_plot(timelines, power, target, args.out)


if __name__ == "__main__":
    main()
