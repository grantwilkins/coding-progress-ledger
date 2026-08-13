"""Plot pooled modeled time to requested power-shed attainment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import network_campaign as campaign
import plot_style
from planner import _expected_scenario
from plot_hardware_shed_frontier import POLICIES, planning_problem
from plot_pooled_shed_frontier import POLICY_STYLE_IDS, pooled_cases, write_csv
from simulate import predict


plot_style.apply()


def attainment_time(commits, shed, target, power_window_s):
    moved = []
    for time_s, session_id in sorted(commits):
        moved.append(session_id)
        if shed(moved) >= target - 1e-8:
            return time_s + power_window_s
    return None


def estimate(plan_paths, requested_fraction=2 / 3, horizon_s=90):
    if not 0 < requested_fraction <= 1 or horizon_s < 30:
        raise ValueError("invalid attainment-CDF horizon or target")
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
        actual = replace(problem, power_limit_w=initial - target)
        for policy, solver in POLICIES.items():
            result = campaign.solve(
                planning_problem(actual, policy), profile, routes, solver,
                seed=scenario["planner_seed"], destination=architecture,
                admission_mode="normal",
            )
            execution = predict(
                _expected_scenario(replace(actual, end_s=horizon_s), result.moves),
                profile, result.moves, destination=architecture,
            )
            commits = [(row.committed_s, row.session_id)
                       for row in execution.sessions
                       if row.committed_s is not None]
            time_s = attainment_time(
                commits,
                lambda moved: initial - campaign.source_power(
                    problem, profile, moved),
                target, profile.power_window_s,
            )
            rows.append({
                "case_id": case_id, "policy": policy,
                "requested_fraction": requested_fraction,
                "deadline_s": 30, "horizon_s": horizon_s,
                "attainment_time_s": "" if time_s is None else time_s,
                "target_met_by_30s": time_s is not None and time_s <= 30,
            })
    return rows


def attainment_curve(rows, policy):
    selected = [row for row in rows if row["policy"] == policy]
    cases = {row["case_id"] for row in selected}
    if len(selected) != len(cases):
        raise RuntimeError("attainment CDF must weight each case once")
    events = sorted(float(row["attainment_time_s"]) for row in selected
                    if row["attainment_time_s"] not in (None, ""))
    return np.r_[0, events], np.r_[0, np.arange(1, len(events) + 1) / len(cases)]


def write_plot(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deadline = 30
    horizon = max(float(row["horizon_s"]) for row in rows)
    fraction = float(rows[0]["requested_fraction"])
    fig, axis = plt.subplots(figsize=plot_style.WIDE_FIGSIZE)
    for policy in POLICIES:
        x, y = attainment_curve(rows, policy)
        axis.step(np.r_[x, horizon], np.r_[y, y[-1]], where="post",
                  **plot_style.policy_style(POLICY_STYLE_IDS[policy]))
    axis.axvline(deadline, color="black", linestyle="--", linewidth=1.5)
    axis.text(
        deadline, .4, "30 s deadline", transform=axis.get_xaxis_transform(),
        ha="center", va="center", rotation=90, fontstyle="italic",
        fontsize=plot_style.LARGE_ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    axis.set(
        xlim=(0, horizon), ylim=(0, 1.02),
        xlabel=f"Time to {fraction:.0%} Power-Shed Attainment (s)",
        ylabel="Cumulative Distribution",
    )
    axis.tick_params(labelsize=plot_style.LARGE_FONT_SIZE)
    axis.xaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    axis.yaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    axis.grid(alpha=.25)
    axis.legend(loc="center right", framealpha=1, facecolor="white",
                edgecolor="none", fontsize=plot_style.LARGE_LEGEND_FONT_SIZE)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requested-fraction", type=float, default=2 / 3)
    parser.add_argument("--horizon-s", type=float, default=90)
    args = parser.parse_args()
    rows = estimate(args.plan, args.requested_fraction, args.horizon_s)
    write_csv(rows, args.out.with_suffix(".csv"))
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
