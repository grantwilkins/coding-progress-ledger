"""Plot distinct views of Queue-Haul action adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import network_campaign as campaign
import plot_style
from plot_hardware_shed_frontier import ACTIONS
from plot_pooled_shed_frontier import pooled_cases, write_csv


POLICY = "queue_haul_lp"
ACTION_MIX_FIGSIZE = (4, 3)
ACTION_MIX_TICK_SIZE = 11
ACTION_MIX_LABEL_SIZE = 12
ACTION_MIX_LEGEND_SIZE = 10
HARDWARE_CASES = (
    "hardware_gap/all-bind", "hardware_gap/free-bandwidth",
    "hardware_gap/free-kv", "hardware_gap/free-service",
    "hardware_gap/all-release",
)
CASE_NAMES = {
    "hardware_gap/all-bind": "All bind",
    "hardware_gap/free-bandwidth": "Free bandwidth",
    "hardware_gap/free-kv": "Free KV",
    "hardware_gap/free-service": "Free service",
    "hardware_gap/all-release": "All released",
    "constraint/window-30": "Unrestricted replay",
    "constraint/quota-30": "Germany replay quota",
}
ACTION_MIX_CASES = (
    ("hardware_gap/free-bandwidth", "HBM + prefill"),
    ("hardware_gap/free-kv", "Bandwidth + prefill"),
    ("hardware_gap/free-service", "Bandwidth + HBM"),
    ("hardware_gap/all-bind", "All bound"),
    ("hardware_gap/all-release", "None bound"),
)
plot_style.apply()


def at_fraction(rows, fraction=2 / 3):
    selected = [row for row in rows if row["policy"] == POLICY and
                abs(float(row["requested_fraction"]) - fraction) < 1e-9]
    cases = [row["case_id"] for row in selected]
    if len(cases) != len(set(cases)) or not selected:
        raise RuntimeError("action view requires one Queue-Haul row per case")
    return selected


def pooled_composition(rows):
    output = []
    fractions = sorted({float(row["requested_fraction"]) for row in rows
                        if row["policy"] == POLICY})
    for fraction in fractions:
        selected = at_fraction(rows, fraction)
        values = {action: np.mean([int(row[action]) / int(row["sessions"])
                                  for row in selected]) for action in ACTIONS}
        values["not_moved"] = np.mean([
            1 - int(row["selected_sessions"]) / int(row["sessions"])
            for row in selected])
        if not np.isclose(sum(values.values()), 1):
            raise RuntimeError("pooled action fractions do not conserve sessions")
        output.append({"requested_fraction": fraction, **values,
                       "cases": len(selected)})
    return output


def controlled_action_mixes(rows, fraction=2 / 3):
    by_case = {row["case_id"]: row for row in at_fraction(rows, fraction)}
    output = []
    for case, label in ACTION_MIX_CASES:
        row = by_case[case]
        selected = int(row["selected_sessions"])
        if int(row["sessions"]) != 28 or selected <= 0 \
                or sum(int(row[action]) for action in ACTIONS) != selected:
            raise RuntimeError("action mix requires an accounted 28-session pack")
        mix = {action: int(row[action]) / selected for action in ACTIONS}
        if not np.isclose(sum(mix.values()), 1):
            raise RuntimeError("selected-action mix does not sum to one")
        output.append({"bound_constraint": label, **mix})
    return output


def _save(fig, path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight" if tight else None)


def _controlled_action_mix(rows, out):
    import matplotlib.pyplot as plt

    labels = {
        "east_replay": "Replay → East", "east_kv_transfer": "KV → East",
        "germany_replay": "Replay → Germany",
        "germany_kv_transfer": "KV → Germany",
    }
    fig, axis = plt.subplots(figsize=ACTION_MIX_FIGSIZE)
    left = np.zeros(len(rows))
    for action in ACTIONS:
        values = np.asarray([row[action] for row in rows]) * 100
        axis.barh(range(len(rows)), values, left=left,
                  color=plot_style.ACTION_COLORS[action],
                  hatch=plot_style.ACTION_HATCHES[action], edgecolor="white",
                  linewidth=1.2, label=labels[action])
        left += values
    axis.set(yticks=range(len(rows)),
             yticklabels=[row["bound_constraint"] for row in rows],
             xlim=(0, 100), xlabel="Selected-action share (%)")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(labelsize=ACTION_MIX_TICK_SIZE)
    axis.xaxis.label.set_size(ACTION_MIX_LABEL_SIZE)
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=2,
               loc="upper right", bbox_to_anchor=(.98, .98),
               fontsize=ACTION_MIX_LEGEND_SIZE, handlelength=1.8,
               columnspacing=.7)
    fig.subplots_adjust(left=.39, right=.95, bottom=.18, top=.68)
    _save(fig, out / "controlled_action_mix", tight=False)
    plt.close(fig)


def _stacked_bars(rows, out):
    import matplotlib.pyplot as plt

    by_case = {row["case_id"]: row for row in rows}
    selected = [by_case[case] for case in HARDWARE_CASES]
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(selected))
    for action in ACTIONS:
        values = np.array([int(row[action]) for row in selected])
        axis.bar(range(len(selected)), values, bottom=bottom,
                 color=plot_style.ACTION_COLORS[action],
                 hatch=plot_style.ACTION_HATCHES[action], edgecolor="white",
                 label=plot_style.ACTION_NAMES[action])
        bottom += values
    for x, row in enumerate(selected):
        axis.text(x, bottom[x] + .5,
                  f"{row['selected_sessions']}/{row['sessions']} moved",
                  ha="center", fontsize=plot_style.ANNOTATION_FONT_SIZE)
    axis.set(xticks=range(len(selected)),
             xticklabels=[CASE_NAMES[row["case_id"]] for row in selected],
             ylabel="Selected sessions", ylim=(0, max(bottom) + 4),
             title="Action mix under controlled resource releases\n"
                   "(67% requested shed)")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, out / "resource_interventions")
    plt.close(fig)


def _quota_dumbbell(rows, out):
    import matplotlib.pyplot as plt

    by_case = {row["case_id"]: row for row in rows}
    base, quota = (by_case[case] for case in
                   ("constraint/window-30", "constraint/quota-30"))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(ACTIONS))
    for i, action in enumerate(ACTIONS):
        a, b = int(base[action]), int(quota[action])
        axis.plot((a, b), (i, i), color="#777777", linewidth=2, zorder=1)
        axis.scatter(a, i, s=90, color=plot_style.ACTION_COLORS[action],
                     marker="o", zorder=2)
        axis.scatter(b, i, s=90, color=plot_style.ACTION_COLORS[action],
                     marker="D", zorder=2)
        axis.text(max(a, b) + .25, i, f"{a} → {b}", va="center",
                  fontsize=plot_style.ANNOTATION_FONT_SIZE)
    axis.set(yticks=y, yticklabels=[plot_style.ACTION_NAMES[a] for a in ACTIONS],
             xlabel="Selected sessions", xlim=(-.5, 8.5),
             title="Response to a Germany replay quota (67% requested shed)")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.scatter([], [], color="#555555", marker="o", label="Unrestricted")
    axis.scatter([], [], color="#555555", marker="D", label="Replay quota")
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out / "replay_quota_response")
    plt.close(fig)


def _demand_area(composition, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    actions = ("replay", "kv_transfer", "not_moved")
    values = {
        "replay": [row["east_replay"] + row["germany_replay"]
                   for row in composition],
        "kv_transfer": [row["east_kv_transfer"] +
                        row["germany_kv_transfer"] for row in composition],
        "not_moved": [row["not_moved"] for row in composition],
    }
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = [row["requested_fraction"] for row in composition]
    axis.stackplot(x, *(values[action] for action in actions),
                   colors=[plot_style.ACTION_COLORS[a] for a in actions],
                   labels=[plot_style.ACTION_NAMES[a] for a in actions],
                   step="mid")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Requested fraction of power",
             ylabel="Mean fraction of source sessions",
             title="Equal-case pooled action composition")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="y", alpha=.2)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out / "pooled_demand_composition")
    plt.close(fig)


def _case_heatmap(rows, out):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    selected = sorted(rows, key=lambda row: row["case_id"])
    matrix = np.array([[int(row[action]) / int(row["sessions"])
                        for action in ACTIONS] for row in selected])
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max(),
                        aspect="auto")
    for i, row in enumerate(selected):
        for j, action in enumerate(ACTIONS):
            axis.text(j, i, row[action], ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                      fontsize=10)
    axis.set(xticks=range(len(ACTIONS)),
             xticklabels=("East\nreplay", "East\nKV transfer",
                          "Germany\nreplay", "Germany\nKV transfer"),
             yticks=range(len(selected)),
             yticklabels=[row["case_id"].replace("_", " ") for row in selected],
             title="Action allocation across all designed cases\n"
                   "(67% requested shed)")
    axis.tick_params(axis="x", labelsize=11)
    colorbar = fig.colorbar(image, ax=axis, pad=.02,
                           label="Fraction of source sessions")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    _save(fig, out / "all_case_action_heatmap")
    plt.close(fig)


def case_metadata(plan_paths):
    metadata = {}
    for path in plan_paths:
        plan = json.loads(path.read_text())
        for scenario in plan["scenarios"]:
            case = f"{plan['design']}/{scenario['condition_id']}"
            metadata.setdefault(case, {
                "bandwidth_gbps": sum(scenario["bandwidth_mbps"].values()) / 1000,
            })
    return metadata


def _bandwidth_response(rows, metadata, out):
    import matplotlib.pyplot as plt

    by_case = {row["case_id"]: row for row in rows}
    cases = ("hardware_gap/all-bind", "hardware_gap/free-bandwidth")
    x = [metadata[case]["bandwidth_gbps"] for case in cases]
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for method, actions in (("Replay", ("east_replay", "germany_replay")),
                            ("KV transfer", ("east_kv_transfer",
                                             "germany_kv_transfer"))):
        y = [sum(int(by_case[case][action]) for action in actions)
             for case in cases]
        color = plot_style.ACTION_COLORS[actions[0]]
        axis.plot(x, y, marker="o", color=color, label=method)
        for a, b, label in zip(x, y, ("Shaped", "Natural")):
            axis.annotate(f"{label}: {b}", (a, b), xytext=(0, 8),
                          textcoords="offset points", ha="center",
                          fontsize=plot_style.ANNOTATION_FONT_SIZE)
    axis.set(xlabel="Combined available bandwidth (Gbit/s)",
             ylabel="Selected sessions", ylim=(0, 12),
             title="Paired response to available bandwidth\n"
                   "(67% requested shed)")
    axis.grid(alpha=.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out / "bandwidth_response")
    plt.close(fig)


def episode_actions(plan_paths, fraction=2 / 3):
    profile = campaign.ModelProfile.load(campaign.MODEL_PATH)
    wanted = set(("hardware_gap/all-bind", "hardware_gap/free-bandwidth",
                  "hardware_gap/all-release"))
    rows = []
    for case, scenario, manifest in pooled_cases(plan_paths):
        if case not in wanted:
            continue
        problem, architecture, routes, _, _ = campaign._scenario_problem(
            scenario, manifest, profile)
        initial = campaign.source_power(problem, profile)
        minimum = campaign.source_power(
            problem, profile, (session.session_id for session in problem.sessions))
        actual = replace(problem, power_limit_w=initial - fraction *
                         (initial - minimum))
        result = campaign.solve(
            actual, profile, routes, "lp_work_first",
            seed=scenario["planner_seed"], destination=architecture,
            admission_mode="normal")
        contexts = {session.session_id: session.context_tokens
                    for session in problem.sessions}
        rows.extend({
            "case_id": case, "order": move.order,
            "context_tokens": contexts[move.session_id],
            "method": move.method,
            "destination": move.destination_instance,
        } for move in result.moves)
    if {row["case_id"] for row in rows} != wanted:
        raise RuntimeError("episode view is missing a requested case")
    return rows


def _episode_scatter(rows, out):
    import matplotlib.pyplot as plt

    cases = ("hardware_gap/all-bind", "hardware_gap/free-bandwidth",
             "hardware_gap/all-release")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, case in zip(axes, cases):
        selected = [row for row in rows if row["case_id"] == case]
        for destination, marker in (("east", "o"), ("germany", "^")):
            for method in ("replay", "kv_transfer"):
                points = [row for row in selected if row["destination"] == destination
                          and row["method"] == method]
                action = f"{destination}_{method}"
                axis.scatter([row["order"] for row in points],
                             [row["context_tokens"] / 1000 for row in points],
                             marker=marker, s=55,
                             color=plot_style.ACTION_COLORS[action],
                             label=f"{method.replace('_', ' ').title()} → {destination.title()}")
        axis.set(title=CASE_NAMES[case], xlabel="Planned migration order")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Session context (k tokens)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(.5, -.04))
    fig.suptitle("Selected sessions within matched modeled episodes "
                 "(67% requested shed)")
    fig.tight_layout(rect=(0, .12, 1, .94))
    _save(fig, out / "matched_episode_actions")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.cases.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = at_fraction(rows)
    composition = pooled_composition(rows)
    mixes = controlled_action_mixes(rows)
    episodes = episode_actions(args.plan)
    write_csv(composition, args.out_dir / "pooled_demand_composition.csv")
    write_csv(episodes, args.out_dir / "matched_episode_actions.csv")
    write_csv(mixes, args.out_dir / "controlled_action_mix.csv")
    _controlled_action_mix(mixes, args.out_dir)
    _stacked_bars(selected, args.out_dir)
    _quota_dumbbell(selected, args.out_dir)
    _demand_area(composition, args.out_dir)
    _case_heatmap(selected, args.out_dir)
    _bandwidth_response(selected, case_metadata(args.plan), args.out_dir)
    _episode_scatter(episodes, args.out_dir)


if __name__ == "__main__":
    main()
