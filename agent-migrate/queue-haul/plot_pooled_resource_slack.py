"""Plot modeled time to bind for pooled resource classes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import network_campaign as campaign
from planner import _expected_scenario
from plot_hardware_shed_frontier import (
    POLICIES, POLICY_COLORS, POLICY_LABELS,
)
from plot_pooled_shed_frontier import pooled_cases, write_csv
from pool_planner import candidate_table
from power_model import ExpectedPower
from simulate import predict


def resource_classes(names):
    classes = {
        "VRAM": tuple(name for name in names if name.startswith("kv:")),
        "Network bandwidth": tuple(name for name in names if
                                   name.startswith("route:") or
                                   name.startswith("migration:") and
                                   name.endswith(":kv_transfer")),
        "Prefill capacity": tuple(name for name in names if
                                  name.startswith("service:") or
                                  name.startswith("migration:") and
                                  name.endswith(":replay")),
    }
    if any(not resources for resources in classes.values()):
        raise RuntimeError("planner table lacks a plotted resource class")
    return classes


def completion_slack(completions, work, classes, deadline_s):
    if deadline_s <= 0 or not set(completions) <= set(work) \
            or any(time < 0 or time > deadline_s for time in completions.values()):
        raise ValueError("invalid completion-ordered resource accounting")
    resources = {resource for members in classes.values() for resource in members}
    if any(value < 0 for row in work.values() for value in row.values()):
        raise ValueError("resource work must be nonnegative")
    used = dict.fromkeys(resources, 0.0)
    rows = [(0.0, dict.fromkeys(classes, 1.0))]
    for session_id, time_s in sorted(
        completions.items(), key=lambda item: (item[1], item[0])):
        for resource in resources:
            used[resource] += work[session_id].get(resource, 0.0)
        slack = {budget: max(0.0, min(1 - used[resource]
                                     for resource in members))
                 for budget, members in classes.items()}
        rows.append((time_s, slack))
    if rows[-1][0] != deadline_s:
        rows.append((deadline_s, dict(rows[-1][1])))
    return rows


def _plan_timeline(problem, architecture, routes, profile, solver, seed,
                   target):
    initial = campaign.source_power(problem, profile)
    planned = replace(problem, power_limit_w=initial - target)
    result = campaign.solve(
        planned, profile, routes, solver, seed=seed, destination=architecture,
        admission_mode="normal",
    )
    table = candidate_table(
        planned, profile, architecture, "normal",
        ExpectedPower(replace(
            planned, final_state="awake", assumed_shutdown_s=None), profile),
    )
    candidates = {}
    for column, candidate in enumerate(table.candidates):
        key = (table.sessions[candidate.session].session_id, candidate.method,
               architecture.pools[candidate.pool].pool_id)
        if key in candidates:
            raise RuntimeError(f"duplicate candidate {key}")
        candidates[key] = column
    work = {}
    for move in result.moves:
        key = move.session_id, move.method, move.destination_pool
        if key not in candidates or move.session_id in work:
            raise RuntimeError(f"missing or duplicate planned candidate {key}")
        column = candidates[key]
        normalized = table.resources[:, column].toarray().ravel()
        work[move.session_id] = dict(zip(table.resource_names, normalized))
    execution = predict(
        _expected_scenario(planned, result.moves), profile, result.moves,
        destination=architecture,
    )
    completions = {row.session_id: row.committed_s for row in execution.sessions
                   if row.committed_s is not None
                   and row.committed_s <= planned.deadline_s}
    return completion_slack(
        completions, work, resource_classes(table.resource_names),
        planned.deadline_s,
    ), initial - result.expected_source_power_at_deadline_w >= target - 1e-8


def estimate(plan_paths, requested_fraction=2 / 3):
    if not 0 < requested_fraction <= 1:
        raise ValueError("requested fraction must lie in (0, 1]")
    profile = campaign.ModelProfile.load(campaign.MODEL_PATH)
    rows = []
    for case_id, scenario, manifest in pooled_cases(plan_paths):
        problem, architecture, routes, _target, _demand = \
            campaign._scenario_problem(scenario, manifest, profile)
        initial = campaign.source_power(problem, profile)
        minimum = campaign.source_power(
            problem, profile,
            (session.session_id for session in problem.sessions))
        target = requested_fraction * (initial - minimum)
        for policy, solver in POLICIES.items():
            timeline, met = _plan_timeline(
                problem, architecture, routes, profile, solver,
                scenario["planner_seed"], target)
            rows.extend({
                "case_id": case_id, "policy": policy, "budget": budget,
                "time_s": time_s, "residual_slack": slack,
                "requested_fraction": requested_fraction,
                "target_met_by_30s": met,
            } for time_s, values in timeline for budget, slack in values.items())
    return rows


def _step_at(rows, grid):
    times = np.asarray([row["time_s"] for row in rows])
    values = np.asarray([row["residual_slack"] for row in rows])
    return values[np.searchsorted(times, grid, side="right") - 1]


def write_plot(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    budgets = ("VRAM", "Network bandwidth", "Prefill capacity")
    policies = tuple(POLICIES)
    cases = sorted({row["case_id"] for row in rows})
    deadline = max(row["time_s"] for row in rows)
    grid = np.linspace(0, deadline, 241)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharex=True, sharey=True)
    for axis, budget in zip(axes, budgets):
        axis.axhspan(0, .05, color="black", alpha=.05)
        for policy in policies:
            trajectories = []
            for case in cases:
                selected = [row for row in rows if row["budget"] == budget
                            and row["policy"] == policy
                            and row["case_id"] == case]
                axis.step(
                    [row["time_s"] for row in selected],
                    [row["residual_slack"] for row in selected], where="post",
                    color=POLICY_COLORS[policy], alpha=.10, linewidth=.8,
                )
                trajectories.append(_step_at(selected, grid))
                bound = next((row for row in selected
                              if row["residual_slack"] <= .05), None)
                if bound:
                    axis.scatter(
                        bound["time_s"], bound["residual_slack"], marker="x",
                        color=POLICY_COLORS[policy], s=12, linewidth=.7,
                    )
            axis.step(
                grid, np.median(trajectories, axis=0), where="post",
                color=POLICY_COLORS[policy], linewidth=2,
            )
        axis.set_title(budget)
        axis.set_xlim(0, deadline)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Episode time (s)")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Residual constraint slack")
    success = {policy: sum(next(
        row["target_met_by_30s"] for row in rows
        if row["case_id"] == case and row["policy"] == policy)
        for case in cases) for policy in policies}
    fig.suptitle(
        f"Modeled time to binding at {rows[0]['requested_fraction']:.0%} requested shed",
        y=.98,
    )
    fig.text(
        .5, .92,
        f"Tightest component per class; thin lines are {len(cases)} cases; "
        "thick lines are medians; × marks first ≤5% slack",
        ha="center", fontsize=9,
    )
    fig.legend(handles=tuple(Line2D(
        [], [], color=POLICY_COLORS[policy], linewidth=2,
        label=f"{POLICY_LABELS[policy]} ({success[policy]}/{len(cases)} targets)",
    ) for policy in policies), frameon=False, ncol=3, loc="lower center",
       bbox_to_anchor=(.5, -.005), fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .86))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requested-fraction", type=float, default=2 / 3)
    args = parser.parse_args()
    rows = estimate(args.plan, args.requested_fraction)
    write_csv(rows, args.out.with_suffix(".csv"))
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
