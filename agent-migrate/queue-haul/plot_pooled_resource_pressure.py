"""Compare normalized planning-budget pressure across designed cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_hardware_shed_frontier import (
    POLICIES, POLICY_COLORS, POLICY_LABELS,
)
from plot_pooled_shed_frontier import write_csv

BUDGETS = {
    "Prefill service": ("service:pool/east:0", "service:pool/germany:0"),
    "KV headroom": ("kv:pool/east", "kv:pool/germany"),
    "Replay window": (
        "migration:pool/east:replay", "migration:pool/germany:replay"),
    "KV-transfer window": (
        "migration:pool/east:kv_transfer",
        "migration:pool/germany:kv_transfer"),
}


def total_utilization(row, resources):
    units = {row[f"unit|{resource}"] for resource in resources}
    capacities = [float(row[f"capacity|{resource}"]) for resource in resources]
    if len(units) != 1 or any(capacity <= 0 for capacity in capacities):
        raise ValueError("total budget requires matching units and positive capacity")
    return sum(float(row[f"used|{resource}"]) for resource in resources) \
        / sum(capacities)


def mean_ci(values, samples=10_000):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError("resource utilization must lie in [0, 1]")
    means = np.random.default_rng(0).choice(
        values, (samples, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(means, (.025, .975))
    return float(values.mean()), float(lower), float(upper)


def summarize(rows, requested_fraction, policies=tuple(POLICIES),
              budgets=BUDGETS):
    selected = [row for row in rows if abs(
        float(row["requested_fraction"]) - requested_fraction) < 1e-9]
    cases = {row["case_id"] for row in selected}
    if not cases:
        raise RuntimeError("requested fraction is absent from the case sweep")
    selected = [{**row, **{budget: total_utilization(row, resources)
                           for budget, resources in budgets.items()}}
                for row in selected]
    summary = []
    for policy in policies:
        policy_rows = [row for row in selected if row["policy"] == policy]
        by_case = {row["case_id"]: row for row in policy_rows}
        if len(policy_rows) != len(cases) or set(by_case) != cases:
            raise RuntimeError("resource summary must weight each case once")
        for budget in budgets:
            mean, lower, upper = mean_ci(
                [row[budget] for row in by_case.values()])
            summary.append({
                "policy": policy, "budget": budget,
                "requested_fraction": requested_fraction,
                "mean": mean, "lower_95": lower, "upper_95": upper,
                "binding_case_fraction": sum(
                    row[budget] >= .95 for row in by_case.values()
                ) / len(cases),
                "cases": len(cases),
            })
    return selected, summary


def write_plot(rows, summary, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    budgets = tuple(BUDGETS)
    policies = tuple(POLICIES)
    cases = sorted({row["case_id"] for row in rows})
    offsets = dict(zip(cases, np.linspace(-.18, .18, len(cases))))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)
    for axis, budget in zip(axes, budgets):
        axis.axhspan(.95, 1, color="black", alpha=.05)
        axis.axhline(1, color="black", linestyle=":", linewidth=1)
        for x, policy in enumerate(policies):
            policy_rows = {row["case_id"]: row for row in rows
                           if row["policy"] == policy}
            for case in cases:
                row = policy_rows[case]
                met = row["target_met_by_30s"] == "True"
                axis.scatter(
                    x + offsets[case], row[budget], s=13,
                    facecolor=POLICY_COLORS[policy] if met else "none",
                    edgecolor=POLICY_COLORS[policy], linewidth=.7, alpha=.65,
                )
            pooled = next(row for row in summary
                          if row["policy"] == policy
                          and row["budget"] == budget)
            axis.errorbar(
                x, pooled["mean"],
                yerr=((pooled["mean"] - pooled["lower_95"],),
                      (pooled["upper_95"] - pooled["mean"],)),
                fmt="D", color=POLICY_COLORS[policy], markeredgecolor="black",
                markersize=5, capsize=3, linewidth=1.5, zorder=3,
            )
        axis.set_title(budget, fontsize=10)
        axis.set_xticks(range(len(policies)),
                        [POLICY_LABELS[policy].replace("Queue-Haul ", "QH ")
                         .replace("Independent-fastest", "Independent")
                         for policy in policies], rotation=30, ha="right",
                        fontsize=8)
        axis.set_ylim(0, 1.06)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Total planning budget used")
    fraction = float(rows[0]["requested_fraction"])
    fig.suptitle(f"Total budget pressure at {fraction:.0%} requested shed", y=.98)
    fig.text(
        .5, .92,
        f"{len(cases)} designed cases; each budget pools all destination capacity; "
        "gray marks ≥95% use",
        ha="center", fontsize=9,
    )
    fig.legend(handles=(
        Line2D([], [], marker="o", linestyle="none", color="black",
               markerfacecolor="black", label="Target met by 30 s"),
        Line2D([], [], marker="o", linestyle="none", color="black",
               markerfacecolor="none", label="Target missed"),
        Line2D([], [], marker="D", linestyle="none", color="black",
               label="Mean and 95% case-bootstrap interval"),
    ), frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.005),
       fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .86))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requested-fraction", type=float, default=2 / 3)
    args = parser.parse_args()
    with args.cases.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected, summary = summarize(rows, args.requested_fraction)
    write_csv(summary, args.out.with_suffix(".csv"))
    write_plot(selected, summary, args.out)


if __name__ == "__main__":
    main()
