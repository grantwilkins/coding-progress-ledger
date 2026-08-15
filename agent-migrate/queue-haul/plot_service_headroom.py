"""Plot the measured A100/H100 incumbent service-headroom curves."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import plot_style
import service_headroom_campaign as campaign


plot_style.apply()


def aggregate(scout: dict) -> list[dict]:
    rows = []
    for direction in plot_style.SERVICE_LOADS:
        selected = [row for row in scout["rows"]
                    if row["direction"] == direction
                    or row["direction"] == "baseline"]
        for rho in sorted({row["target_rho"] for row in selected}):
            block = [row for row in selected if row["target_rho"] == rho]
            values = {
                metric: [row[metric] for row in block if row[metric] is not None]
                for metric in ("p90_ttft_s", "p90_mean_tpot_s")
            }
            rows.append({
                "hardware": scout["hardware"], "direction": direction,
                "target_rho": rho, "measured_rho_median": statistics.median(
                    row["offered_rho"] for row in block),
                "blocks": len(block),
                "physically_feasible": all(row["stable"] for row in block),
                **{f"{metric}_median": statistics.median(metric_values)
                   if metric_values else None
                   for metric, metric_values in values.items()},
                **{f"{metric}_minimum": min(metric_values)
                   if metric_values else None
                   for metric, metric_values in values.items()},
                **{f"{metric}_maximum": max(metric_values)
                   if metric_values else None
                   for metric, metric_values in values.items()},
            })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], scout: dict, confirmed: dict | None,
         out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metrics = (("p90_ttft_s", "Incumbent P90 TTFT (s)"),
               ("p90_mean_tpot_s", "Incumbent P90 mean TPOT (s)"))
    figure, axes = plt.subplots(2, 1, sharex=True,
                                figsize=plot_style.COMPACT_FIGSIZE)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for direction in plot_style.SERVICE_LOADS:
            selected = [row for row in rows if row["direction"] == direction
                        and row[f"{metric}_median"] is not None]
            x = [row["measured_rho_median"] for row in selected]
            y = [row[f"{metric}_median"] for row in selected]
            lower = [row[f"{metric}_median"] - row[f"{metric}_minimum"]
                     for row in selected]
            upper = [row[f"{metric}_maximum"] - row[f"{metric}_median"]
                     for row in selected]
            axis.plot(x, y, color=plot_style.SERVICE_LOAD_COLORS[direction],
                      linestyle=plot_style.SERVICE_LOAD_LINESTYLES[direction],
                      label=plot_style.SERVICE_LOAD_NAMES[direction])
            feasible = [index for index, row in enumerate(selected)
                        if row["physically_feasible"]]
            infeasible = [index for index, row in enumerate(selected)
                          if not row["physically_feasible"]]
            if feasible:
                axis.errorbar([x[index] for index in feasible],
                              [y[index] for index in feasible],
                              yerr=([lower[index] for index in feasible],
                                    [upper[index] for index in feasible]),
                              color=plot_style.SERVICE_LOAD_COLORS[direction],
                              marker=plot_style.SERVICE_LOAD_MARKERS[direction],
                              linestyle="none", capsize=2, markersize=5)
            if infeasible:
                axis.scatter([x[index] for index in infeasible],
                             [y[index] for index in infeasible], marker="x",
                             color=plot_style.SERVICE_LOAD_COLORS[direction],
                             s=45, linewidths=1.5)
        target = scout["targets"][metric]
        axis.axhline(target, color="#555555", linestyle=":", linewidth=1.5)
        if confirmed and confirmed.get("planner_usable"):
            axis.axvline(confirmed["supported_bound"], color="#777777",
                         linestyle="-.", linewidth=1.25)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel(r"Measured total normalized service load ($\rho$)")
    handles = [Line2D([], [],
                      color=plot_style.SERVICE_LOAD_COLORS[direction],
                      linestyle=plot_style.SERVICE_LOAD_LINESTYLES[direction],
                      marker=plot_style.SERVICE_LOAD_MARKERS[direction],
                      label=plot_style.SERVICE_LOAD_NAMES[direction])
               for direction in plot_style.SERVICE_LOADS]
    handles.append(Line2D([], [], color="#555555", linestyle=":",
                          label="Evaluation target"))
    handles.append(Line2D([], [], color="#555555", marker="x",
                          linestyle="none",
                          label="Unstable, undrained, or failed"))
    if confirmed and confirmed.get("planner_usable"):
        handles.append(Line2D([], [], color="#777777", linestyle="-.",
                              label=r"Confirmed $\rho_{safe}$"))
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out.with_suffix(f".{suffix}"),
                       dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--confirmed", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.confirmed) != bool(args.confirmation_plan):
        parser.error("--confirmed and --confirmation-plan must be supplied together")
    plan = campaign.read_plan(args.plan)
    scout = json.loads(args.scout.read_text())
    campaign.validate_scout_evidence(plan, scout, scout["hardware"])
    confirmed = json.loads(args.confirmed.read_text()) if args.confirmed else None
    if confirmed:
        confirmation_plan = campaign.read_plan(args.confirmation_plan)
        campaign.supported_bound(confirmed, confirmation_plan, plan, scout)
    rows = aggregate(scout)
    write_csv(rows, args.out.with_suffix(".csv"))
    plot(rows, scout, confirmed, args.out)


if __name__ == "__main__":
    main()
