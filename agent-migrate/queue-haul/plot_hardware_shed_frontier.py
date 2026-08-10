"""Sweep requested shed on the frozen all-bind hardware scenario."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import network_campaign as campaign
from plot_hardware_constraint_timeline import _resolve


POLICIES = {
    "queue_haul_lp": "lp_work_first",
    "queue_haul_greedy": "greedy",
    "independent_fastest": "isolated_fastest",
    "replay_only": "replay_only",
    "kv_only": "kv_only",
    "power_blind": "lp_power_blind",
}
RESOURCES = (
    "kv:pool/east",
    "service:pool/germany:0",
    "migration:pool/east:replay",
    "migration:pool/east:kv_transfer",
    "migration:pool/germany:replay",
    "migration:pool/germany:kv_transfer",
)
ACTIONS = (
    "east_replay", "east_kv_transfer",
    "germany_replay", "germany_kv_transfer",
)


def plateau_attainment(requested, safe_shed, admissible):
    """Cap overshed at the request and retain the last safe result on failure."""
    if not (len(requested) == len(safe_shed) == len(admissible)) \
            or any(b < a for a, b in zip(requested, requested[1:])):
        raise ValueError("frontier inputs must be aligned and request-sorted")
    frontier, last = [], 0.0
    for target, shed, safe in zip(requested, safe_shed, admissible):
        if target < 0 or shed < 0:
            raise ValueError("shed values must be nonnegative")
        if safe:
            last = max(last, min(target, shed))
        frontier.append(last)
    return frontier


def sweep(plan_path: Path, points: int = 41):
    if points < 2:
        raise ValueError("frontier requires at least two requested-shed points")
    plan = json.loads(plan_path.read_text())
    manifest = json.loads(_resolve(plan["manifest"]["path"], plan_path).read_text())
    profile = campaign.ModelProfile.load(campaign.MODEL_PATH)
    scenario = next(row for row in plan["scenarios"]
                    if row["condition_id"] == "all-bind"
                    and row["repeat"] == 0
                    and row["policy"] == "queue_haul_robust")
    problem, architecture, routes, hardware_target, _demand = \
        campaign._scenario_problem(scenario, manifest, profile)
    initial = campaign.source_power(problem, profile)
    minimum = campaign.source_power(
        problem, profile, (session.session_id for session in problem.sessions))
    requested = sorted(set(np.linspace(0, initial - minimum, points))
                       | {hardware_target})
    rows = []
    for policy, solver in POLICIES.items():
        raw, admissible, policy_rows = [], [], []
        for target in requested:
            if target == 0:
                result = None
                safe_shed, resources, counts = 0.0, {}, dict.fromkeys(ACTIONS, 0)
                safe, failure, selected, bottleneck = True, None, 0, None
            else:
                result = campaign.solve(
                    replace(problem, power_limit_w=initial - target),
                    profile, routes, solver, seed=scenario["planner_seed"],
                    destination=architecture, admission_mode="normal",
                )
                safe_shed = max(0.0, initial - result.expected_source_power_at_deadline_w)
                resources = {row.name: row.utilization for row in result.resource_uses}
                max_use = max(resources.values(), default=0)
                safe = max_use <= 1 + 1e-8 and result.failure_reason in {None, "target_unmet"}
                failure, selected, bottleneck = (
                    result.failure_reason, len(result.moves), result.bottleneck)
                counts = campaign._constraint_action_counts(result.moves)
            raw.append(safe_shed)
            admissible.append(safe)
            policy_rows.append({
                "requested_shed_w": target,
                "policy": policy,
                "raw_safe_shed_w": safe_shed,
                "plan_safe": safe,
                "target_met_by_30s": safe and safe_shed >= target - 1e-8,
                "failure_reason": failure,
                "selected_sessions": selected,
                "bottleneck": bottleneck,
                **{resource: resources.get(resource, 0) for resource in RESOURCES},
                **{action: counts[action] for action in ACTIONS},
            })
        attained = plateau_attainment(requested, raw, admissible)
        rows.extend({**row, "safely_attained_shed_w": value}
                    for row, value in zip(policy_rows, attained))
    return rows, hardware_target


def write_csv(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, hardware_target: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "queue_haul_lp": "Queue-Haul LP",
        "queue_haul_greedy": "Queue-Haul Greedy",
        "independent_fastest": "Independent-fastest",
        "replay_only": "Replay-only",
        "kv_only": "KV-only",
        "power_blind": "Power-blind",
    }
    colors = dict(zip(POLICIES, (
        campaign.TAB10_COLORS[0], campaign.TAB10_COLORS[1],
        campaign.TAB10_COLORS[6], campaign.TAB10_COLORS[3],
        campaign.TAB10_COLORS[2], campaign.TAB10_COLORS[7],
    )))
    resource_labels = (
        "East KV", "Germany service", "East replay", "East KV transfer",
        "Germany replay", "Germany KV transfer",
    )
    action_labels = (
        "Replay → East", "KV → East", "Replay → Germany", "KV → Germany",
    )
    fig = plt.figure(figsize=(14, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1),
                            height_ratios=(1.1, .75), hspace=.32, wspace=.28)
    frontier = fig.add_subplot(grid[:, 0])
    heat = fig.add_subplot(grid[0, 1])
    mix = fig.add_subplot(grid[1, 1], sharex=heat)
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        frontier.plot(
            [row["requested_shed_w"] for row in selected],
            [row["safely_attained_shed_w"] for row in selected],
            color=colors[policy], linewidth=2, label=labels[policy],
        )
    maximum = max(row["requested_shed_w"] for row in rows)
    frontier.plot((0, maximum), (0, maximum), color="black", linestyle=":",
                  linewidth=1, label="Requested = attained")
    frontier.axvline(
        hardware_target, color="black", linestyle="--", linewidth=1,
        label=f"Hardware request ({hardware_target:.2f} W)",
    )
    frontier.set(xlim=(0, maximum), ylim=(0, maximum),
                 xlabel="Requested shed (W)",
                 ylabel="Safely attained shed by 30 s (W)",
                 title="All-bind attainment frontier")
    frontier.grid(alpha=.2)
    frontier.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    selected = [row for row in rows if row["policy"] == "queue_haul_lp"]
    x = np.asarray([row["requested_shed_w"] for row in selected])
    pressure = np.asarray([[row[resource] for row in selected]
                           for resource in RESOURCES])
    image = heat.imshow(
        pressure, vmin=0, vmax=1, aspect="auto", cmap="viridis",
        extent=(x[0], x[-1], len(RESOURCES) - .5, -.5),
        interpolation="nearest",
    )
    for column, row in enumerate(np.argmax(pressure, axis=0)):
        if pressure[row, column] >= .95:
            heat.text(x[column], row, "×", ha="center", va="center",
                      color="white", fontsize=8)
    heat.set_yticks(range(len(RESOURCES)), resource_labels, fontsize=8)
    heat.axvline(hardware_target, color="black", linestyle="--", linewidth=1)
    heat.set(title="Queue-Haul LP resource pressure (× ≥ 95%)")
    heat.tick_params(axis="x", labelbottom=False)
    fig.colorbar(image, ax=heat, label="Budget used", fraction=.045, pad=.02)

    shares = np.asarray([[row[action] / row["selected_sessions"]
                          if row["selected_sessions"] else 0
                          for row in selected] for action in ACTIONS])
    mix.stackplot(
        x, shares, labels=action_labels,
        colors=campaign.TAB10_COLORS[:4], step="mid",
    )
    mix.axvline(hardware_target, color="black", linestyle="--", linewidth=1)
    mix.set(xlim=(0, maximum), ylim=(0, 1), xlabel="Requested shed (W)",
            ylabel="Selected-action share",
            title="Queue-Haul LP action/destination mix")
    mix.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    mix.grid(axis="y", alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=41)
    args = parser.parse_args()
    rows, target = sweep(args.plan, args.points)
    write_csv(rows, args.out)
    write_plot(rows, target, args.out)


if __name__ == "__main__":
    main()
