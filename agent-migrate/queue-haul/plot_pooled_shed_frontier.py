"""Pool normalized 30-second shed frontiers across designed cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import network_campaign as campaign
from plot_hardware_constraint_timeline import _resolve
from plot_hardware_shed_frontier import (
    POLICIES, POLICY_COLORS, POLICY_LABELS, sweep_scenario,
)


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


def write_plot(summary, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    fig, axis = plt.subplots(figsize=(8, 6))
    for policy in POLICIES:
        selected = [row for row in summary if row["policy"] == policy]
        x = [row["requested_fraction"] for row in selected]
        axis.fill_between(
            x, [row["lower_quartile"] for row in selected],
            [row["upper_quartile"] for row in selected],
            color=POLICY_COLORS[policy], alpha=.12, linewidth=0,
        )
        axis.plot(
            x, [row["median"] for row in selected],
            color=POLICY_COLORS[policy], linewidth=2,
            label=POLICY_LABELS[policy],
        )
    axis.plot((0, 1), (0, 1), color="black", linestyle=":", linewidth=1,
              label="Requested = attained")
    axis.set(xlim=(0, 1), ylim=(0, 1),
             xlabel="Requested fraction of removable power",
             ylabel="Safely attained fraction by 30 s",
             title=f"Pooled 30 s attainment frontier ({summary[0]['cases']} cases)")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="gray", alpha=.2))
    labels.append("Interquartile case range")
    axis.legend(handles, labels, frameon=False, fontsize=8, ncol=2,
                loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=31)
    args = parser.parse_args()
    rows = sweep(args.plan, args.points)
    summary = pooled_summary(rows)
    write_csv(rows, args.out.with_name(f"{args.out.name}_cases.csv"))
    write_csv(summary, args.out.with_suffix(".csv"))
    write_plot(summary, args.out)


if __name__ == "__main__":
    main()
