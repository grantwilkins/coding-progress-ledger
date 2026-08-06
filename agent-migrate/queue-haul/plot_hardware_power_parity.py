"""Plot requested versus achieved power shed across hardware campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy_hardware_campaign import deadline_attainment
from profiles import ModelProfile


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
POLICY_ROOTS = (
    *(OUTPUTS / f"policy-hardware-width8-{name}-20260730"
      for name in ("frontier", "packing")),
    OUTPUTS / "policy-hardware-width8-isolated-fastest-20260806",
)
NETWORK_ROOT = OUTPUTS / "network-campaign-20260805"
METHODS = ("queue_haul", "greedy", "greedy_lagrangian", "isolated_fastest",
           "kv_only", "replay_only", "random")
PLOT_METHODS = ("queue_haul", "greedy", "isolated_fastest", "kv_only",
                "replay_only")
TARGET_TOLERANCE_PP = 5
METHOD_LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "greedy_lagrangian": "Lagrangian greedy",
    "isolated_fastest": "Per-session fastest", "kv_only": "KV only",
    "replay_only": "Replay only", "random": "Random",
}


def _point(campaign: str, scenario: str, method: str, requested: float,
           achieved: float) -> dict:
    if method not in METHODS or not 0 <= requested <= 100 \
            or not 0 <= achieved <= 100:
        raise ValueError("invalid normalized power-shed point")
    return {
        "campaign": campaign, "scenario_id": scenario, "method": method,
        "requested_percent": requested, "achieved_percent": achieved,
        "marker": "x" if achieved < requested - TARGET_TOLERANCE_PP else "o",
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


def outcomes(rows: list[dict]) -> list[tuple[str, int, float]]:
    counts = {"Below target": 0, "On target": 0, "Above target": 0}
    for row in rows:
        error = row["achieved_percent"] - row["requested_percent"]
        counts["Below target" if error < -TARGET_TOLERANCE_PP else
               "Above target" if error > TARGET_TOLERANCE_PP
               else "On target"] += 1
    return [(name, count, 100 * count / len(rows))
            for name, count in counts.items()]


def method_outcomes(rows: list[dict]) -> list[tuple[str, list[tuple]]]:
    summaries = [(method, outcomes(selected)) for method in PLOT_METHODS
                 if (selected := [row for row in rows
                                  if row["method"] == method])]
    return sorted(summaries,
                  key=lambda row: row[1][1][2] + row[1][2][2], reverse=True)


def write_plot(rows: list[dict], out: Path) -> None:
    summaries = method_outcomes(rows)
    colors = ("#B1040E", "#006CB8", "#008566")
    fig, axis = plt.subplots(figsize=(6, 3))
    for y, (method, summary) in enumerate(summaries):
        left = 0
        for (name, _count, percent), color in zip(summary, colors):
            axis.barh(y, percent, left=left, height=.62, color=color,
                      label=name if y == 0 else None)
            if percent >= 8:
                axis.text(left + percent / 2, y, f"{percent:.1f}%",
                          color="white", ha="center", va="center",
                          weight="bold")
            left += percent
        met = summary[1][2] + summary[2][2]
        axis.text(101, y, f"{met:.1f}% met", va="center", fontsize=9)
    axis.set(xlim=(0, 114), xticks=range(0, 101, 20),
             yticks=range(len(summaries)),
             yticklabels=[METHOD_LABELS[method] for method, _ in summaries],
             xlabel="Share of hardware scenarios (%)")
    axis.invert_yaxis()
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="x", alpha=.2, zorder=0)
    axis.legend(frameon=False, ncol=3, loc="lower center",
                bbox_to_anchor=(.5, 1.01))
    fig.tight_layout()
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
