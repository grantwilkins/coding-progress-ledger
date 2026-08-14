"""Plot the paired live-repair and repair-disabled hardware result."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import migration_profiler as profiler
import plot_style


ROOT = Path(__file__).parent
DEFAULT_PLAN = ROOT / "outputs/repair-disabled-control-20260814/plan.json"
DEFAULT_REPAIRED_ROOT = Path("/datadrive/queue-haul-repair-20260814-r3")
DEFAULT_CONTROL_ROOT = Path(
    "/datadrive/queue-haul-repair-disabled-control-20260814-r1")
DEFAULT_OUT = ROOT / "outputs/repair-control-comparison-20260814"
CONTROL_SCHEMA = "queue-haul-scheduled-repair-disabled-control-plan-v1"
ACTIONS = (
    "east_replay", "east_kv_transfer",
    "germany_replay", "germany_kv_transfer", "not_moved",
)

plot_style.apply()


def _assignment_rows(result: dict) -> list[dict]:
    return sorted(({
        "session_id": row["session_id"],
        "destination": row["assignment"]["destination"],
        "method": row["assignment"]["method"],
    } for row in result["repair_result"]["moves"]),
        key=lambda row: row["session_id"])


def _validate_result(result: dict, outcome: str) -> None:
    requests = result.get("requests", ())
    if result.get("status") != "complete" \
            or result.get("repair_outcome") != outcome \
            or not result.get("shadow_guard", {}).get("passed") \
            or not result.get("repair_result", {}).get("reaches_target") \
            or not requests \
            or any(row.get("ttft_s") is None
                   or row.get("request", {}).get("status_code") != 200
                   for row in requests):
        raise RuntimeError(f"invalid paired {outcome} hardware result")


def load_pairs(plan_path: Path, repaired_root: Path,
               control_root: Path) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    repaired_validation = json.loads(
        (repaired_root / "validation.json").read_text())
    control_validation = json.loads(
        (control_root / "validation.json").read_text())
    if plan.get("schema") != CONTROL_SCHEMA \
            or not repaired_validation.get("passed") \
            or not control_validation.get("passed"):
        raise RuntimeError("paired repair/control evidence did not validate")
    pairs = []
    for episode in sorted(plan["episodes"], key=lambda row: row["repeat"]):
        control = json.loads((control_root / "episodes" /
                              episode["episode_id"] / "result.json").read_text())
        repaired = json.loads((repaired_root / "episodes" /
                               episode["paired_repair_episode_id"] /
                               "result.json").read_text())
        _validate_result(control, "disabled")
        _validate_result(repaired, "applied")
        if profiler.object_hash(control["initial_moves"]) \
                != episode["expected_initial_moves_sha256"] \
                or profiler.object_hash(repaired["initial_moves"]) \
                != episode["expected_initial_moves_sha256"] \
                or _assignment_rows(control) != _assignment_rows(repaired):
            raise RuntimeError("paired run changed its initial plan or repair")
        pairs.append({"episode": episode, "control": control,
                      "repaired": repaired})
    if len(pairs) != 3:
        raise RuntimeError("paired comparison requires exactly three repeats")
    return pairs


def action_mix(result: dict) -> dict[str, int]:
    initial = {row["session_id"] for row in result["initial_moves"]}
    executed = {row["session_id"] for row in result["requests"]}
    if not executed <= initial:
        raise RuntimeError("executed action was absent from the initial plan")
    counts = Counter(
        f"{row['destination_instance']}_{row['method']}"
        for row in result["requests"])
    counts["not_moved"] = len(initial - executed)
    output = {action: counts[action] for action in ACTIONS}
    if sum(output.values()) != len(initial):
        raise RuntimeError("action disposition does not conserve sessions")
    return output


def _metric(result: dict, percentile: float) -> float:
    values = sorted(row["ttft_s"] for row in result["requests"])
    return values[int(percentile * (len(values) - 1))]


def comparison_summary(pairs: list[dict]) -> dict:
    mixes = {
        "replan": [action_mix(pair["repaired"]) for pair in pairs],
        "no_replan": [action_mix(pair["control"]) for pair in pairs],
    }
    if any(rows[1:] != rows[:-1] for rows in mixes.values()):
        raise RuntimeError("action disposition changed across paired repeats")

    def values(policy: str, field: str) -> list[float]:
        key = "repaired" if policy == "replan" else "control"
        return [float(pair[key][field]) for pair in pairs]

    def ttfts(policy: str, percentile: float | None) -> list[float]:
        key = "repaired" if policy == "replan" else "control"
        return [max(row["ttft_s"] for row in pair[key]["requests"])
                if percentile is None else _metric(pair[key], percentile)
                for pair in pairs]

    metrics = {}
    for policy in plot_style.SCHEDULE_COMPARISON_NAMES:
        metrics[policy] = {
            "time_to_target_s": values(policy, "time_to_target_s"),
            "deadline_shed_w": values(policy, "realized_shed_w"),
            "executed_actions": [
                len(pair["repaired" if policy == "replan" else "control"][
                    "requests"]) for pair in pairs],
            "ttft_p50_s": ttfts(policy, .5),
            "ttft_p90_s": ttfts(policy, .9),
            "ttft_max_s": ttfts(policy, None),
        }
        metrics[policy]["means"] = {
            name: statistics.mean(rows) for name, rows in metrics[policy].items()
        }
    representative = pairs[0]
    return {
        "schema": "queue-haul-repair-control-comparison-v1",
        "repeats": len(pairs),
        "representative": {
            "control_episode_id": representative["control"]["episode_id"],
            "repaired_episode_id": representative["repaired"]["episode_id"],
            "event_s": representative["repaired"]["event_s"],
            "decision_s": representative["repaired"]["decision_s"],
            "requested_shed_w": representative["repaired"]["requested_shed_w"],
        },
        "action_mix": {policy: rows[0] for policy, rows in mixes.items()},
        "metrics": metrics,
    }


def _plot_action_mix(axis, summary: dict) -> None:
    policies = ("replan", "no_replan")
    left = [0, 0]
    for action in ACTIONS:
        values = [summary["action_mix"][policy][action] for policy in policies]
        bars = axis.barh(
            range(len(policies)), values, left=left,
            color=plot_style.ACTION_COLORS[action],
            hatch=plot_style.ACTION_HATCHES.get(action, ""),
            edgecolor="white", linewidth=1.1,
        )
        for bar, value in zip(bars, values):
            if value:
                axis.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_y() + bar.get_height() / 2, str(value),
                          ha="center", va="center", fontsize=10,
                          color="white" if action in {
                              "germany_replay", "germany_kv_transfer"} else
                          "black")
        left = [old + value for old, value in zip(left, values)]
    axis.set(
        yticks=range(len(policies)),
        yticklabels=[plot_style.SCHEDULE_COMPARISON_NAMES[value]
                     for value in policies],
        xlim=(0, 15), xticks=range(0, 16, 3),
        xlabel="Disposition of the 15 initially planned sessions",
        title="(a) QH changes the pending action mix",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.18)
    handles = [Patch(
        facecolor=plot_style.ACTION_COLORS[action],
        hatch=plot_style.ACTION_HATCHES.get(action, ""), edgecolor="white",
        label=plot_style.ACTION_NAMES[action]) for action in ACTIONS]
    axis.legend(handles=handles, frameon=False, ncol=3,
                loc="upper center", bbox_to_anchor=(.5, -.36), fontsize=8.5,
                columnspacing=1.0, handlelength=1.8)


def _curve(result: dict, horizon: float) -> tuple[list[float], list[float]]:
    rows = result["attainment_curve"]
    times = [float(row["time_s"]) for row in rows]
    watts = [float(row["shed_w"]) for row in rows]
    if times[-1] < horizon:
        times.append(horizon)
        watts.append(watts[-1])
    return times, watts


def _plot_attainment(axis, pair: dict, summary: dict) -> None:
    results = {"replan": pair["repaired"], "no_replan": pair["control"]}
    horizon = max(row["attainment_curve"][-1]["time_s"]
                  for row in results.values())
    for policy, result in results.items():
        times, watts = _curve(result, horizon)
        axis.step(
            times, watts, where="post",
            color=plot_style.SCHEDULE_COMPARISON_COLORS[policy],
            linestyle=plot_style.SCHEDULE_COMPARISON_LINESTYLES[policy],
            label=plot_style.SCHEDULE_COMPARISON_NAMES[policy],
        )
    target = summary["representative"]["requested_shed_w"]
    event = summary["representative"]["event_s"]
    decision = summary["representative"]["decision_s"]
    axis.axhline(
        target, color=plot_style.EVENT_COLORS["shed_target"],
        linestyle=plot_style.EVENT_LINESTYLES["shed_target"], linewidth=1.5,
    )
    axis.axvline(
        event, color=plot_style.EVENT_COLORS["resource_shift"],
        linestyle=plot_style.EVENT_LINESTYLES["resource_shift"], linewidth=1.8)
    axis.set(
        xlim=(0, horizon + 3), ylim=(0, 53),
        xlabel="Time from migration start (s)",
        ylabel="Cumulative source power shed (W)",
        title="(b) Replanning avoids the long tail",
    )
    axis.grid(alpha=.18)
    axis.text(horizon, results["replan"]["attainment_curve"][-1]["shed_w"] + .8,
              plot_style.SCHEDULE_COMPARISON_NAMES["replan"],
              ha="right", va="bottom", fontsize=9,
              color=plot_style.SCHEDULE_COMPARISON_COLORS["replan"])
    axis.text(horizon * .38, target - .8, f"Target: {target:.1f} W",
              ha="left", va="top", fontsize=8.5,
              color=plot_style.EVENT_COLORS["shed_target"])
    axis.annotate(
        f"10× Germany bandwidth + prefill drop at {event:.1f} s",
        xy=(event, 49), xytext=(12, 49), textcoords="data",
        arrowprops={"arrowstyle": "->",
                    "color": plot_style.EVENT_COLORS["resource_shift"]},
        color=plot_style.EVENT_COLORS["resource_shift"], fontsize=8.5,
        va="center")
    control = results["no_replan"]
    last = max(control["attainment_curve"], key=lambda row: row["time_s"])
    axis.annotate(
        f"No replan: 15th action at {last['time_s']:.0f} s\n"
        f"max TTFT = {max(row['ttft_s'] for row in control['requests']):.1f} s",
        xy=(last["time_s"], last["shed_w"]),
        xytext=(horizon - 70, 43),
        arrowprops={"arrowstyle": "->",
                    "color": plot_style.SCHEDULE_COMPARISON_COLORS["no_replan"]},
        color=plot_style.SCHEDULE_COMPARISON_COLORS["no_replan"], fontsize=9)

    inset = axis.inset_axes([.43, .08, .52, .40])
    for policy, result in results.items():
        times, watts = _curve(result, 30)
        inset.step(
            times, watts, where="post",
            color=plot_style.SCHEDULE_COMPARISON_COLORS[policy],
            linestyle=plot_style.SCHEDULE_COMPARISON_LINESTYLES[policy],
            linewidth=2,
        )
        inset.scatter(
            [result["time_to_target_s"]], [target], s=30, zorder=4,
            color=plot_style.SCHEDULE_COMPARISON_COLORS[policy])
    inset.axhline(target, color=plot_style.EVENT_COLORS["shed_target"],
                  linestyle=plot_style.EVENT_LINESTYLES["shed_target"],
                  linewidth=1)
    inset.axvline(event, color=plot_style.EVENT_COLORS["resource_shift"],
                  linestyle=plot_style.EVENT_LINESTYLES["resource_shift"],
                  linewidth=1)
    inset.axvline(decision, color=plot_style.EVENT_COLORS["repair_decision"],
                  linestyle=plot_style.EVENT_LINESTYLES["repair_decision"],
                  linewidth=1)
    inset.set(xlim=(0, 28), ylim=(0, 40))
    inset.set_title("Target attainment (first 28 s)", fontsize=8.5, pad=2)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=.15)
    inset.text(event - .25, 39, f"drop {event:.1f} s", rotation=90,
               ha="right", va="top",
               fontsize=7.5,
               color=plot_style.EVENT_COLORS["resource_shift"])
    inset.text(decision + .25, 39, f"replan {decision:.1f} s", rotation=90,
               ha="left", va="top",
               fontsize=7.5,
               color=plot_style.EVENT_COLORS["repair_decision"])
    repaired = results["replan"]
    inset.text(repaired["time_to_target_s"] - .5, target + 1.4,
               f"{repaired['time_to_target_s']:.1f} s", ha="right",
               color=plot_style.SCHEDULE_COMPARISON_COLORS["replan"],
               fontsize=8)
    inset.text(control["time_to_target_s"] + .5, target - 3.7,
               f"{control['time_to_target_s']:.1f} s", ha="left",
               color=plot_style.SCHEDULE_COMPARISON_COLORS["no_replan"],
               fontsize=8)


def run(plan_path: Path, repaired_root: Path,
        control_root: Path, out: Path) -> dict:
    pairs = load_pairs(plan_path, repaired_root, control_root)
    summary = comparison_summary(pairs)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "summary.json", summary)
    mix_rows = [{"policy": policy, "action": action, "count": count}
                for policy, mix in summary["action_mix"].items()
                for action, count in mix.items()]
    profiler.write_csv(out / "action_mix.csv", mix_rows)
    representative = pairs[0]
    curve_rows = [{"policy": policy, **row}
                  for policy, result in (
                      ("replan", representative["repaired"]),
                      ("no_replan", representative["control"]))
                  for row in result["attainment_curve"]]
    profiler.write_csv(out / "attainment.csv", curve_rows)

    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 6.5),
        gridspec_kw={"height_ratios": (.85, 1.7)})
    _plot_action_mix(axes[0], summary)
    _plot_attainment(axes[1], representative, summary)
    for axis in axes:
        axis.title.set_fontsize(13)
    fig.subplots_adjust(left=.17, right=.98, bottom=.10, top=.94, hspace=.88)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"repair_control_comparison.{suffix}",
                    dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repaired-root", type=Path,
                        default=DEFAULT_REPAIRED_ROOT)
    parser.add_argument("--control-root", type=Path,
                        default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run(
        args.plan, args.repaired_root, args.control_root, args.out), indent=2))


if __name__ == "__main__":
    main()
