"""Plot requested versus achieved power shed across hardware campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from policy_hardware_campaign import deadline_attainment
from profiles import ModelProfile


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
POLICY_ROOTS = tuple(OUTPUTS / f"policy-hardware-width8-{name}-20260730"
                     for name in ("frontier", "packing"))
NETWORK_ROOT = OUTPUTS / "network-campaign-20260805"
METHODS = ("queue_haul", "greedy", "greedy_lagrangian", "kv_only",
           "replay_only", "random")
LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Greedy",
    "greedy_lagrangian": "Lagrangian greedy", "kv_only": "KV only",
    "replay_only": "Replay only", "random": "Random",
}
COLORS = dict(zip(METHODS, ("#B1040E", "#008566", "#620059", "#006CB8",
                            "#E98300", "#767676")))


def _point(campaign: str, scenario: str, method: str, requested: float,
           achieved: float) -> dict:
    if method not in METHODS or not 0 <= requested <= 100 \
            or not 0 <= achieved <= 100:
        raise ValueError("invalid normalized power-shed point")
    return {
        "campaign": campaign, "scenario_id": scenario, "method": method,
        "requested_percent": requested, "achieved_percent": achieved,
        "marker": "x" if achieved < requested - 1e-9 else "o",
    }


def _requested(scenario: dict, moves: list[dict]) -> float:
    sessions = len(scenario["sessions"])
    if sessions != 8 or {move["session_id"] for move in moves} != {
            row["session_id"] for row in scenario["sessions"]}:
        raise ValueError("power parity requires complete width-8 decisions")
    return 100 * sum(move["deadline_admitted"] for move in moves) / sessions


def load_policy(root: Path) -> list[dict]:
    plan = json.loads((root / "plan.json").read_text())
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]
                 if row["policy"] != "control"}
    with (root / "policy_attainment.csv").open() as handle:
        attainment = list(csv.DictReader(handle))
    if {row["scenario_id"] for row in attainment} != set(scenarios):
        raise ValueError(f"attainment does not cover the policy plan: {root}")
    return [
        _point(root.name, row["scenario_id"], row["policy"],
               _requested(scenarios[row["scenario_id"]],
                          scenarios[row["scenario_id"]]["moves"]),
               100 * float(row["power_attainment_fraction"]))
        for row in attainment
    ]


def load_network(root: Path, power_curve, power_window_s: float) -> list[dict]:
    plan = json.loads((root / "plan.json").read_text())
    points = []
    for scenario in plan["scenarios"]:
        attempts = sorted((root / "scenarios" / scenario["scenario_id"])
                          .glob("attempt-*/result.json"))
        if not attempts:
            continue
        result = json.loads(attempts[-1].read_text())
        if result.get("status") != "complete":
            continue
        decision = json.loads(
            (attempts[-1].parent / "decision.json").read_text())["moves"]
        commits = [(row["request"]["end_ns"] - result["started_ns"]) / 1e9
                   for row in result["requests"]]
        achieved = deadline_attainment(
            commits, len(scenario["sessions"]), [scenario["deadline_s"]],
            power_curve, power_window_s,
        )[0]["power_attainment_fraction"]
        points.append(_point(
            root.name, scenario["scenario_id"], scenario["policy"],
            _requested(scenario, decision), 100 * achieved,
        ))
    return points


def load_handoff(root: Path, idle_power_w: float) -> dict:
    result = json.loads((root / "result.json").read_text())
    with (root / "power_summary.csv").open() as handle:
        power = {(row["node"], row["phase"]): float(row["mean_power_w"])
                 for row in csv.DictReader(handle)}
    pre, post = power["sweden", "pre"], power["sweden", "post"]
    if pre <= idle_power_w:
        raise ValueError("handoff baseline must exceed model-resident idle power")
    achieved = min(100, max(0, 100 * (pre - post) / (pre - idle_power_w)))
    return _point(root.name, result["scenario"]["scenario_id"],
                  result["scenario"]["policy"],
                  _requested(result["scenario"], result["decision"]["moves"]),
                  achieved)


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["requested_percent"]), []).append(
            row["achieved_percent"])
    output = []
    for (method, requested), values in sorted(groups.items()):
        low, median, high = np.percentile(values, (10, 50, 90))
        output.append({
            "method": method, "requested_percent": requested,
            "achieved_percent": median, "low_percent": low,
            "high_percent": high, "samples": len(values),
            "marker": "x" if median < requested - 1e-9 else "o",
        })
    return output


def write_plot(rows: list[dict], out: Path) -> None:
    summaries = summarize(rows)
    methods = [method for method in METHODS
               if any(row["method"] == method for row in summaries)]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, methods):
        selected = [row for row in summaries if row["method"] == method]
        for row in selected:
            x, y = row["requested_percent"], row["achieved_percent"]
            axis.errorbar(
                x, y, yerr=((y - row["low_percent"],),
                            (row["high_percent"] - y,)),
                fmt="none", color=COLORS[method], capsize=3, lw=1.4,
            )
            axis.plot((x, x), (x, y), color=COLORS[method], alpha=.35, lw=1)
            axis.scatter(x, y, color=COLORS[method], marker=row["marker"],
                         s=42, linewidths=1.4, clip_on=False, zorder=3)
        axis.plot((0, 100), (0, 100), color="black", ls="--", lw=.8)
        axis.set_title(f"{LABELS[method]} (n={sum(row['samples'] for row in selected)})",
                       color=COLORS[method], fontsize=11)
        axis.set(xlim=(0, 100), ylim=(0, 100))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Requested shed (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Achieved shed (%)")
    fig.legend(handles=(
        Line2D([], [], marker="x", ls="", color="black",
               label="Median below request"),
        Line2D([], [], marker="o", ls="", color="black",
               label="Median at/above request"),
        Line2D([], [], marker="|", ls="-", color="black",
               label="10th–90th percentile"),
    ), frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, .06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=OUTPUTS / "hardware_power_shed_parity")
    args = parser.parse_args()
    model = ModelProfile.load(ROOT / "profiles/gpt_oss_20b_a100_tp1.json")
    rows = [row for root in POLICY_ROOTS for row in load_policy(root)]
    joint = NETWORK_ROOT / "joint-queue-002-partial-086"
    rows += load_network(joint, model.case().power_curve, model.power_window_s)
    rows += [load_handoff(NETWORK_ROOT / name, model.case().power_curve.power(0))
             for name in ("handoff", "handoff-010")]
    write_plot(rows, args.out)


if __name__ == "__main__":
    main()
