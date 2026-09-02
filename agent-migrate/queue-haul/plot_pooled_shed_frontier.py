"""Pool normalized 30-second shed frontiers across designed cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import network_campaign as campaign
import plot_style
from plot_hardware_constraint_timeline import _resolve
from plot_hardware_shed_frontier import (
    POLICIES, sweep_scenario,
)


POLICY_STYLE_IDS = dict(zip(POLICIES, (
    "queue_haul", "greedy", "isolated_fastest", "replay_only", "kv_only",
    "queue_haul_power_blind", "queue_haul_deadline_blind",
)))
plot_style.apply()


def pooled_summary(rows):
    summary = []
    cases = {row["case_id"] for row in rows}
    for policy in POLICIES:
        fractions = sorted({row["requested_fraction"] for row in rows
                            if row["policy"] == policy})
        for fraction in fractions:
            selected = [row for row in rows if row["policy"] == policy
                        and row["requested_fraction"] == fraction]
            by_case = {row["case_id"]: row["safely_attained_fraction"]
                       for row in selected}
            if len(selected) != len(cases) or set(by_case) != cases:
                raise RuntimeError("pooled frontier does not weight each case once")
            lower, median, upper = np.quantile(list(by_case.values()),
                                               (.25, .5, .75))
            summary.append({
                "policy": policy, "requested_fraction": fraction,
                "lower_quartile": lower, "median": median,
                "upper_quartile": upper, "cases": len(cases),
            })
    return summary


def pooled_cases(plan_paths: list[Path]):
    case_ids = set()
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text())
        manifest = json.loads(_resolve(plan["manifest"]["path"], plan_path).read_text())
        for condition in sorted({row["condition_id"] for row in plan["scenarios"]}):
            candidates = [row for row in plan["scenarios"]
                          if row["condition_id"] == condition
                          and row["repeat"] == 0]
            by_policy = {row["policy"]: row for row in candidates}
            template = next(by_policy[policy] for policy in (
                "queue_haul", "queue_haul_robust") if policy in by_policy)
            case_id = f"{plan['design']}/{condition}"
            if case_id in case_ids:
                raise RuntimeError(f"duplicate pooled case {case_id}")
            case_ids.add(case_id)
            scenario = {**template, "deadline_s": 30, "planning_deadline_s": 30}
            yield case_id, scenario, manifest


def sweep(plan_paths: list[Path], points: int):
    if points < 2:
        raise ValueError("pooled frontier requires at least two points")
    profile = campaign.ModelProfile.load(campaign.MODEL_PATH)
    fractions, rows = np.linspace(0, 1, points), []
    for case_id, scenario, manifest in pooled_cases(plan_paths):
        rows.extend(sweep_scenario(
            scenario, manifest, profile, fractions, case_id))
    return rows


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hardware_point(path: Path, requested_fraction: float) -> dict:
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["policy"] == "queue_haul_robust"]
    cases = {(row["condition_index"], row["repeat"]) for row in rows}
    if len(rows) != 15 or len(cases) != 15 or any(
            row["status"] != "complete" or not row["deadline_met"] == "True"
            for row in rows):
        raise RuntimeError("H100 hardware point requires 15 complete matched runs")
    attained = [requested_fraction * float(row["realized_shed_w"])
                / float(row["requested_shed_w"]) for row in rows]
    return {"requested_fraction": requested_fraction,
            "attained_fraction": float(np.median(attained)),
            "observations": len(rows)}


def queue_haul_cutoff(summary) -> float:
    rows = sorted((row for row in summary if row["policy"] == "queue_haul_lp"),
                  key=lambda row: row["requested_fraction"])
    if len(rows) < 2:
        raise RuntimeError("Queue-Haul frontier requires at least two points")
    crossings = []
    for left, right in zip(rows, rows[1:]):
        x0, x1 = left["requested_fraction"], right["requested_fraction"]
        d0, d1 = left["median"] - x0, right["median"] - x1
        if x0 > 0 and d0 == 0:
            crossings.append(x0)
        if d0 > 0 >= d1:
            crossings.append(x0 + (x1 - x0) * d0 / (d0 - d1))
    if not crossings:
        raise RuntimeError("Queue-Haul median frontier does not cross parity")
    return max(crossings)


def write_plot(summary, out: Path, measured=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    cutoff = queue_haul_cutoff(summary)
    fig, axis = plt.subplots(figsize=(4, 3))
    for policy in POLICIES:
        selected = [row for row in summary if row["policy"] == policy]
        x = [row["requested_fraction"] for row in selected]
        axis.fill_between(
            x, [row["lower_quartile"] for row in selected],
            [row["upper_quartile"] for row in selected],
            color=plot_style.POLICY_COLORS[POLICY_STYLE_IDS[policy]],
            alpha=.12, linewidth=0,
        )
        axis.plot(
            x, [row["median"] for row in selected],
            **plot_style.policy_style(POLICY_STYLE_IDS[policy]),
        )
    axis.plot((0, cutoff), (0, cutoff), color="black", linestyle=":",
              linewidth=1)
    if measured:
        axis.scatter(measured["requested_fraction"],
                     measured["attained_fraction"], marker="*", s=110,
                     color=plot_style.POLICY_COLORS["queue_haul"],
                     edgecolor="black", linewidth=.8, zorder=5,
                     label="H100 measured Queue-Haul")
    axis.set(xlim=(0, .8), ylim=(0, 1),
             xlabel="Requested Source-Power Fraction",
             ylabel="Source Power Shed")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=1,
               loc="center left", bbox_to_anchor=(.94, .5))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--hardware-results", type=Path)
    args = parser.parse_args()
    rows = sweep(args.plan, args.points)
    summary = pooled_summary(rows)
    write_csv(rows, args.out.with_name(f"{args.out.name}_cases.csv"))
    write_csv(summary, args.out.with_suffix(".csv"))
    measured = hardware_point(args.hardware_results, .414) \
        if args.hardware_results else None
    write_plot(summary, args.out, measured)


if __name__ == "__main__":
    main()
