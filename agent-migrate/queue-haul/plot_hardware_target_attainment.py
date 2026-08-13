"""Plot uncapped hardware shed relative to each requested target."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from plot_hardware_shed_frontier import POLICY_COLORS, POLICY_LABELS


POLICIES = {
    "queue_haul": "queue_haul_lp",
    "greedy": "queue_haul_greedy",
    "isolated_fastest": "independent_fastest",
    "replay_only": "replay_only",
    "kv_only": "kv_only",
    "queue_haul_power_blind": "power_blind",
}


def summarize(rows, scenarios):
    fractions = {}
    for scenario in scenarios:
        key, value = int(scenario["condition_index"]), float(
            scenario["requested_shed_fraction"])
        if key in fractions and fractions[key] != value:
            raise RuntimeError("condition has conflicting requested fractions")
        fractions[key] = value
    rows = [row for row in rows if row["policy"] in POLICIES]
    if any(row["status"] != "complete" or int(row["request_failures"])
           or row["deadline_met"] != "True" for row in rows):
        raise RuntimeError("hardware target attainment requires deadline-safe requests")
    active = {row["policy"] for row in rows}
    summary = []
    for policy in POLICIES:
        if policy not in active:
            continue
        for condition, fraction in fractions.items():
            values = [float(row["realized_shed_w"]) /
                      float(row["requested_shed_w"]) for row in rows
                      if row["policy"] == policy
                      and int(row["condition_index"]) == condition]
            if not values:
                raise RuntimeError("hardware target attainment lacks a policy cell")
            lower, median, upper = np.quantile(values, (.25, .5, .75))
            summary.append({
                "policy": policy, "requested_fraction": fraction,
                "lower_quartile": lower, "median": median,
                "upper_quartile": upper, "episodes": len(values),
            })
    return summary


def write_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(5, 4))
    for policy, display in POLICIES.items():
        selected = sorted((row for row in rows if row["policy"] == policy),
                          key=lambda row: row["requested_fraction"])
        x = [row["requested_fraction"] for row in selected]
        axis.fill_between(
            x, [row["lower_quartile"] for row in selected],
            [row["upper_quartile"] for row in selected],
            color=POLICY_COLORS[display], alpha=.12, linewidth=0)
        axis.plot(x, [row["median"] for row in selected], marker="o",
                  color=POLICY_COLORS[display], linewidth=2, markersize=4,
                  label=POLICY_LABELS[display])
    axis.axhline(1, color="black", linestyle=":", linewidth=1)
    axis.set(xlim=(.5, .8), ylim=(.4, 1.75),
             xlabel="Requested Fraction of Power",
             ylabel="Fraction of Requested Power\nAttained by Deadline")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.grid(alpha=.2)
    axis.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.results.open() as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize(rows, json.loads(args.plan.read_text())["scenarios"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary, args.out.with_suffix(".csv"))
    write_plot(summary, args.out)


if __name__ == "__main__":
    main()
