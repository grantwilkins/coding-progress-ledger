"""Plot 2xA100 load and bandwidth power-shed operating curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from profiles import ModelProfile


ROOT = Path(__file__).parent
LOAD_ROOT = ROOT / "outputs/capacity-load-publication-20260807"
BANDWIDTH_ROOT = ROOT / "outputs/policy-hardware-width8-packing-20260730"
FULL_DRAIN_ROOTS = tuple(
    ROOT / f"outputs/capacity-full-drain-block{block}-20260807"
    for block in range(2)
)
OUT = ROOT / "outputs/capacity-operating-curves-20260808"
SCHEDULERS = ("queue_haul", "greedy", "replay_only", "kv_only")
QH_SCHEDULERS = SCHEDULERS[:2]
LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "replay_only": "Replay only", "kv_only": "KV only",
}
COLORS = {
    "queue_haul": "#B1040E", "greedy": "#008566",
    "replay_only": "#E98300", "kv_only": "#006CB8",
}
MARKERS = dict(zip(SCHEDULERS, ("o", "s", "^", "D")))
LINESTYLES = dict(zip(SCHEDULERS, ("-", "--", "-.", ":")))
ACTIONS = ("replay", "kv_transfer", "not_moved")
ACTION_LABELS = ("Replay", "KV transfer", "Not moved")
ACTION_COLORS = ("#E98300", "#006CB8", "#999999")
FIELDS = (
    "campaign", "split", "deadline_s", "scheduler", "independent_value",
    "observations", "time_to_full_power_s", "deadline_attainment_fraction",
    "watts_shed_by_deadline", "target_w", "replay_action_fraction",
    "kv_transfer_action_fraction", "not_moved_action_fraction",
)


def _csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _profile(record: dict) -> ModelProfile:
    saved = Path(record["path"])
    path = saved if saved.is_file() else ROOT / "profiles" / saved.name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() \
            != record["sha256"]:
        raise RuntimeError("campaign model profile is unavailable or changed")
    return ModelProfile.load(path)


def action_fractions(scenarios: list[dict]) -> dict[str, float]:
    counts, total = Counter(), 0
    for scenario in scenarios:
        sessions = {row["session_id"] for row in scenario["sessions"]}
        moves = [row["session_id"] for row in scenario["moves"]]
        if not sessions or len(moves) != len(set(moves)) \
                or not set(moves) <= sessions:
            raise ValueError("actions must uniquely cover a subset of sessions")
        methods = Counter(row["method"] for row in scenario["moves"])
        if set(methods) - set(ACTIONS[:2]):
            raise ValueError("unknown migration action")
        counts.update(methods)
        counts["not_moved"] += len(sessions) - len(moves)
        total += len(sessions)
    if not total:
        raise ValueError("action fractions need scenarios")
    return {action: counts[action] / total for action in ACTIONS}


def _row(campaign, split, deadline, scheduler, x, cells, scenarios,
         times, watts, target):
    actions = action_fractions(scenarios)
    return {
        "campaign": campaign, "split": split, "deadline_s": deadline,
        "scheduler": "queue_haul" if scheduler == "lp" else scheduler,
        "independent_value": x, "observations": len(cells),
        "time_to_full_power_s": statistics.median(times) if times else None,
        "deadline_attainment_fraction":
            sum(time <= deadline for time in times) / len(cells),
        "watts_shed_by_deadline": statistics.median(watts), "target_w": target,
        **{f"{action}_action_fraction": value
           for action, value in actions.items()},
    }


def summarize_load(rows: list[dict], scenarios: list[dict],
                   power_window_s: float) -> list[dict]:
    plans = {row["scenario_id"]: row for row in scenarios}
    if len(plans) != len(scenarios) or {row["scenario_id"] for row in rows} \
            != set(plans):
        raise ValueError("load results and plan do not match")
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["load_fraction"]), row["policy"]].append(row)
    output = []
    for (load, scheduler), cells in sorted(groups.items()):
        selected = [plans[row["scenario_id"]] for row in cells]
        deadlines = {float(row["required_deadline_s"]) for row in selected}
        targets = {float(row["requested_shed_w"]) for row in cells}
        if len(deadlines) != 1 or len(targets) != 1:
            raise ValueError("load cell mixes deadlines or targets")
        target, times = targets.pop(), []
        for result, scenario in zip(cells, selected):
            sessions = len(scenario["sessions"])
            if int(result["planned_sessions"]) == sessions \
                    and int(result["credited_sessions"]) == sessions \
                    and float(result["achieved_shed_w"]) >= target - 1e-6:
                times.append(float(result["full_drain_s"]) + power_window_s)
        output.append(_row(
            "load", "controlled_destination_load", deadlines.pop(), scheduler,
            statistics.median(float(row["offered_rho"]) for row in cells),
            cells, selected, times,
            [float(row["achieved_shed_w"]) for row in cells], target,
        ))
    return output


def bandwidth_observations(plan: dict, episodes: list[dict],
                           attainment: list[dict], target_w: float,
                           power_window_s: float) -> list[dict]:
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]
                 if row["policy"] != "control"}
    summaries = {row["scenario_id"]: row for row in episodes}
    achieved = {row["scenario_id"]: row for row in attainment}
    if not scenarios or set(scenarios) != set(summaries) or set(scenarios) \
            != set(achieved):
        raise ValueError("bandwidth plan, episodes, and attainment do not match")
    output = []
    for identifier, scenario in scenarios.items():
        summary, result = summaries[identifier], achieved[identifier]
        complete = summary["commit_100_s"] not in (None, "") \
            and int(summary["planned_migrations"]) == len(scenario["sessions"]) \
            and int(summary["completed_migrations"]) == len(scenario["sessions"])
        output.append({
            "scenario_id": identifier,
            "deadline_s": float(scenario["required_deadline_s"]),
            "split": scenario["context_profile"],
            "scheduler": scenario["policy"],
            "independent_value": float(scenario["bandwidth_mbps"]) / 1000,
            "time_to_full_power_s":
                float(summary["commit_100_s"]) + power_window_s if complete else None,
            "watts_shed_by_deadline":
                target_w * float(result["power_attainment_fraction"]),
            "target_w": target_w,
        })
    profiles = {deadline: {row["split"] for row in output
                           if row["deadline_s"] == deadline}
                for deadline in {row["deadline_s"] for row in output}}
    for key in {(row["deadline_s"], row["independent_value"], row["scheduler"])
                for row in output}:
        counts = Counter(row["split"] for row in output
                         if (row["deadline_s"], row["independent_value"],
                             row["scheduler"]) == key)
        if set(counts) != profiles[key[0]] or len(set(counts.values())) != 1:
            raise ValueError("pooled context profiles are not balanced")
    return output


def summarize_bandwidth(plan: dict, episodes: list[dict], attainment: list[dict],
                        target_w: float, power_window_s: float) -> list[dict]:
    observations = bandwidth_observations(
        plan, episodes, attainment, target_w, power_window_s)
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]
                 if row["policy"] != "control"}
    groups = defaultdict(list)
    for row in observations:
        groups[row["deadline_s"], row["split"], row["independent_value"],
               row["scheduler"]].append(row)
    output = []
    for (deadline, profile, bandwidth, scheduler), cells in sorted(groups.items()):
        selected = [scenarios[row["scenario_id"]] for row in cells]
        output.append(_row(
            "bandwidth", profile, deadline, scheduler, bandwidth, cells, selected,
            [row["time_to_full_power_s"] for row in cells
             if row["time_to_full_power_s"] is not None],
            [row["watts_shed_by_deadline"] for row in cells], target_w,
        ))
    return output


def summarize_full_drain(rows: list[dict], scenarios: list[dict],
                         power_window_s: float) -> list[dict]:
    plans = {row["scenario_id"]: row for row in scenarios}
    if len(plans) != len(scenarios) or {row["scenario_id"] for row in rows} \
            != set(plans):
        raise ValueError("full-drain results and plans do not match")
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["load_fraction"]),
               float(row["configured_goodput_mbps"]), row["policy"]].append(row)
    output = []
    for (load, bandwidth, scheduler), cells in sorted(groups.items()):
        selected = [plans[row["scenario_id"]] for row in cells]
        deadlines = {float(row["required_deadline_s"]) for row in selected}
        targets = {float(row["requested_shed_w"]) for row in cells}
        if len(deadlines) != 1 or len(targets) != 1 or any(
                int(row["planned_sessions"]) != len(scenario["sessions"])
                for row, scenario in zip(cells, selected)):
            raise ValueError("full-drain cell mixes contracts or omits sessions")
        target = targets.pop()
        output.append(_row(
            "bandwidth_full_drain", f"{load:g} load", deadlines.pop(),
            scheduler, bandwidth / 1000, cells, selected,
            [float(row["full_drain_s"]) + power_window_s for row in cells],
            [float(row["achieved_shed_w"]) for row in cells], target,
        ))
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scheduler_plot(rows, field, ylabel, out, xlabel, splits):
    columns = min(3, len(splits))
    plot_rows = math.ceil(len(splits) / columns)
    figure, axes = plt.subplots(
        plot_rows, columns,
        figsize=(6.4, 4.2) if len(splits) == 1 else (10.5, 3.1 * plot_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    deadline = rows[0]["deadline_s"]
    for axis, split in zip(axes.flat, splits):
        panel = [row for row in rows if row["split"] == split]
        visible = [scheduler for scheduler in SCHEDULERS
                   if any(row["scheduler"] == scheduler for row in panel)]
        axis.axhline(deadline if field == "time_to_full_power_s"
                     else panel[0]["target_w"], color="black", linestyle="--",
                     zorder=0)
        for scheduler in SCHEDULERS:
            points = sorted((row for row in panel
                             if row["scheduler"] == scheduler),
                            key=lambda row: row["independent_value"])
            if not points:
                continue
            values = [row["independent_value"] for row in points]
            span = max(values) - min(values) or 1
            offset = span * .003 * (
                visible.index(scheduler) - (len(visible) - 1) / 2)
            x = [value + offset for value in values]
            axis.plot(
                x,
                [float("nan") if row[field] is None else row[field]
                for row in points], color=COLORS[scheduler],
                marker=MARKERS[scheduler], linestyle=LINESTYLES[scheduler],
                label=LABELS[scheduler], zorder=2,
            )
            if field == "time_to_full_power_s":
                missing = [index for index, row in enumerate(points)
                           if row[field] is None]
                axis.scatter([x[index] for index in missing],
                             [deadline] * len(missing), marker="x", s=45,
                             color=COLORS[scheduler], zorder=3)
        axis.set_title(split.replace("_", " ").title())
        axis.grid(alpha=.25)
    for axis in axes.flat[len(splits):]:
        axis.set_visible(False)
    axes.flat[0].set_ylabel(ylabel)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel(xlabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2,
                  loc="upper center", bbox_to_anchor=(.5, .99))
    figure.text(.995, .005, "Scheduler curves offset slightly for visibility",
                ha="right", fontsize=7, color="#666666")
    figure.tight_layout(rect=(0, .03, 1, .84 if len(splits) == 1 else .9))
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)


def _action_plot(rows, out, xlabel, splits):
    figure, axes = plt.subplots(
        1 if len(splits) == 1 else 2, 2 if len(splits) == 1 else len(splits),
        figsize=(8, 3.6) if len(splits) == 1 else (15, 5.8),
        sharex=True, sharey=True, squeeze=False,
    )
    for row_index, scheduler in enumerate(QH_SCHEDULERS):
        for column, split in enumerate(splits):
            axis = axes[0, row_index] if len(splits) == 1 else axes[row_index, column]
            points = sorted((row for row in rows if row["split"] == split
                             and row["scheduler"] == scheduler),
                            key=lambda row: row["independent_value"])
            for action, label, color in zip(ACTIONS, ACTION_LABELS, ACTION_COLORS):
                axis.plot([row["independent_value"] for row in points],
                          [row[f"{action}_action_fraction"] for row in points],
                          marker="o", label=label, color=color)
            axis.set_ylim(-.02, 1.02)
            axis.grid(alpha=.25)
            axis.set_title(LABELS[scheduler] if len(splits) == 1
                           else split.title())
            if column == 0:
                axis.set_ylabel("Action fraction" if len(splits) == 1 else
                                f"{LABELS[scheduler]}\nAction fraction")
            if row_index == (0 if len(splits) == 1 else 1):
                axis.set_xlabel(xlabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3,
                  loc="upper center", bbox_to_anchor=(.5, 1.02))
    if splits == ("pooled_contexts",):
        figure.text(.5, .005, "Pooled equally across five context profiles",
                    ha="center", fontsize=8, color="#666666")
    figure.tight_layout(rect=(0, .03 if splits == ("pooled_contexts",) else 0,
                              1, .92))
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)


def _cdf(values, total):
    ordered = sorted(value for value in values if value is not None)
    return [0, *ordered], [0, *(index / total
                                for index in range(1, len(ordered) + 1))]


def _bandwidth_cdf(rows, field, xlabel, out):
    bandwidths = sorted({row["independent_value"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True,
                                sharey=True, squeeze=False)
    deadline = rows[0]["deadline_s"]
    for axis, bandwidth in zip(axes.flat, bandwidths):
        panel = [row for row in rows if row["independent_value"] == bandwidth]
        axis.axvline(deadline if field == "time_to_full_power_s"
                     else panel[0]["target_w"], color="black", linestyle="--",
                     zorder=0)
        for scheduler in SCHEDULERS:
            selected = [row[field] for row in panel
                        if row["scheduler"] == scheduler]
            x, y = _cdf(selected, len(selected))
            axis.step(x, y, where="post", color=COLORS[scheduler],
                      linestyle=LINESTYLES[scheduler], linewidth=2,
                      label=LABELS[scheduler], zorder=2)
        axis.set_title(f"{bandwidth:g} Gbit/s")
        axis.grid(alpha=.25)
        axis.set_xlim(left=0)
    for axis in axes[:, 0]:
        axis.set_ylabel("Cumulative fraction")
    for axis in axes[-1]:
        axis.set_xlabel(xlabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2,
                  loc="upper center", bbox_to_anchor=(.5, .99))
    figure.text(.5, .005,
                "Five context profiles × three repeats = 15 episodes per curve",
                ha="center", fontsize=8, color="#666666")
    figure.tight_layout(rect=(0, .035, 1, .88))
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)


def _pooled_actions(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_s"], row["scheduler"],
               row["independent_value"]].append(row)
    output = []
    for (deadline, scheduler, value), cells in sorted(groups.items()):
        total = sum(row["observations"] for row in cells)
        output.append({
            **cells[0], "split": "pooled_contexts", "observations": total,
            **{f"{action}_action_fraction": sum(
                row[f"{action}_action_fraction"] * row["observations"]
                for row in cells) / total for action in ACTIONS},
        })
    return output


def write(load_root: Path = LOAD_ROOT, bandwidth_root: Path = BANDWIDTH_ROOT,
          full_drain_roots=FULL_DRAIN_ROOTS, out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    load_plan = json.loads((load_root / "live_plan.json").read_text())
    load_model = _profile(load_plan["profile"])
    load = summarize_load(_csv(load_root / "live_capacity.csv"),
                          load_plan["scenarios"], load_model.power_window_s)
    bandwidth_plan = json.loads((bandwidth_root / "plan.json").read_text())
    bandwidth_model = _profile(bandwidth_plan["model_profile"])
    curve = bandwidth_model.case().power_curve
    target = curve.power(.4) - curve.power(0)
    bandwidth_episodes = _csv(bandwidth_root / "policy_episodes.csv")
    bandwidth_attainment = _csv(bandwidth_root / "policy_attainment.csv")
    bandwidth = summarize_bandwidth(
        bandwidth_plan, bandwidth_episodes, bandwidth_attainment, target,
        bandwidth_model.power_window_s)
    bandwidth_observed = bandwidth_observations(
        bandwidth_plan, bandwidth_episodes, bandwidth_attainment, target,
        bandwidth_model.power_window_s)
    full_drain_plans = [json.loads((root / "live_plan.json").read_text())
                        for root in full_drain_roots]
    full_drain = summarize_full_drain(
        [row for root in full_drain_roots
         for row in _csv(root / "full_drain_capacity.csv")],
        [row for plan in full_drain_plans for row in plan["scenarios"]],
        _profile(full_drain_plans[0]["profile"]).power_window_s,
    )
    _write_csv(out / "load_curves.csv", load)
    _write_csv(out / "bandwidth_curves.csv", bandwidth)
    _write_csv(out / "full_drain_bandwidth_curves.csv", full_drain)
    _scheduler_plot(load, "time_to_full_power_s", "Time to full power shed (s)",
                    out / "load_time_to_full_power", "Normalized offered load",
                    ("controlled_destination_load",))
    _scheduler_plot(load, "watts_shed_by_deadline", "Watts shed by 30 s (W)",
                    out / "load_watts_by_30s", "Normalized offered load",
                    ("controlled_destination_load",))
    _action_plot(load, out / "load_action_selection",
                 "Normalized offered load", ("controlled_destination_load",))
    full_drain_splits = tuple(sorted(
        {row["split"] for row in full_drain}, key=lambda value: float(value.split()[0])
    ))
    _scheduler_plot(
        full_drain, "time_to_full_power_s", "Time to full power shed (s)",
        out / "full_drain_bandwidth_time_to_full_power", "Bandwidth (Gbit/s)",
        full_drain_splits,
    )
    _scheduler_plot(
        full_drain, "watts_shed_by_deadline", "Watts shed by 30 s (W)",
        out / "full_drain_bandwidth_watts_by_30s", "Bandwidth (Gbit/s)",
        full_drain_splits,
    )
    for deadline in sorted({row["deadline_s"] for row in bandwidth}):
        selected = [row for row in bandwidth if row["deadline_s"] == deadline]
        observed = [row for row in bandwidth_observed
                    if row["deadline_s"] == deadline]
        splits = tuple(sorted({row["split"] for row in selected}))
        prefix = out / f"bandwidth_{deadline:g}s"
        _scheduler_plot(
            selected, "time_to_full_power_s", "Time to full power shed (s)",
            Path(f"{prefix}_time_to_full_power"), "Bandwidth (Gbit/s)", splits,
        )
        _scheduler_plot(
            selected, "watts_shed_by_deadline",
            f"Watts shed by {deadline:g} s (W)",
            Path(f"{prefix}_watts_by_deadline"), "Bandwidth (Gbit/s)", splits,
        )
        _action_plot(selected, Path(f"{prefix}_action_selection"),
                     "Bandwidth (Gbit/s)", splits)
        _bandwidth_cdf(
            observed, "time_to_full_power_s", "Time to full power shed (s)",
            Path(f"{prefix}_time_to_full_power_cdf"),
        )
        _bandwidth_cdf(
            observed, "watts_shed_by_deadline",
            f"Watts shed by {deadline:g} s (W)",
            Path(f"{prefix}_watts_by_deadline_cdf"),
        )
        _action_plot(
            [row for row in _pooled_actions(selected)
             if row["deadline_s"] == deadline],
            Path(f"{prefix}_action_selection_pooled"), "Bandwidth (Gbit/s)",
            ("pooled_contexts",),
        )
    return load, bandwidth, full_drain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-root", type=Path, default=LOAD_ROOT)
    parser.add_argument("--bandwidth-root", type=Path, default=BANDWIDTH_ROOT)
    parser.add_argument("--full-drain-root", type=Path, nargs="+",
                        default=FULL_DRAIN_ROOTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    write(args.load_root, args.bandwidth_root, args.full_drain_root, args.out)


if __name__ == "__main__":
    main()
