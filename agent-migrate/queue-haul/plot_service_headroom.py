"""Plot the measured A100/H100 incumbent service-headroom curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import plot_style
import service_headroom_campaign as campaign


plot_style.apply()


def phase_coordinates(plan: dict, rates: dict, row: dict) -> dict[str, float]:
    trace = campaign.offered_trace(
        plan, rates, row["direction"], row["target_rho"], row["block"],
    )
    phases = campaign.offered_phase_rho(plan, trace)
    if not math.isclose(sum(phases.values()), row["offered_rho"],
                        rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("phase coordinates do not reproduce offered rho")
    return phases


def _aggregate(hardware: str, raw_rows: list[dict], directions: tuple[str, ...],
               *, stage: str, targets: dict | None = None,
               plan: dict | None = None, rates: dict | None = None) -> list[dict]:
    rows = []
    if bool(plan) != bool(rates):
        raise ValueError("plan and normalization must be supplied together")
    for direction in directions:
        selected = [row for row in raw_rows
                    if row["direction"] == direction
                    or row["direction"] == "baseline"]
        for rho in sorted({row["target_rho"] for row in selected}):
            block = [row for row in selected if row["target_rho"] == rho]
            phases = [phase_coordinates(plan, rates, row) for row in block] \
                if plan and rates else []
            values = {
                metric: [row[metric] for row in block if row[metric] is not None]
                for metric in ("p90_ttft_s", "p90_mean_tpot_s")
            }
            rows.append({
                "hardware": hardware, "stage": stage, "direction": direction,
                "target_rho": rho, "measured_rho_median": statistics.median(
                    row["offered_rho"] for row in block),
                "offered_prefill_rho_median": statistics.median(
                    row["offered_prefill_rho"] for row in phases)
                if phases else None,
                "offered_decode_rho_median": statistics.median(
                    row["offered_decode_rho"] for row in phases)
                if phases else None,
                "blocks": len(block),
                "physically_feasible": all(row["stable"] for row in block),
                "evidence_feasible": all(campaign.row_feasible(row, targets)
                                         for row in block)
                if targets else all(row["stable"] for row in block),
                "feasible_blocks": sum(campaign.row_feasible(row, targets)
                                       for row in block)
                if targets else sum(row["stable"] for row in block),
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


def aggregate(scout: dict, plan: dict | None = None,
              rates: dict | None = None) -> list[dict]:
    return _aggregate(
        scout["hardware"], scout["rows"], plot_style.SERVICE_LOADS,
        stage="discovery", targets=scout.get("targets"), plan=plan, rates=rates,
    )


def aggregate_confirmation(result: dict, plan: dict, rates: dict) -> list[dict]:
    return _aggregate(
        result["hardware"], result["rows"], plot_style.SERVICE_MIXES,
        stage="held_out", targets=result["targets"], plan=plan, rates=rates,
    )


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

    metrics = (("p90_ttft_s", "P90 TTFT (s, log scale)"),
               ("p90_mean_tpot_s", "P90 mean TPOT (s)"))
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
        if metric == "p90_ttft_s":
            axis.set_yscale("log")
        axis.set_ylabel(ylabel, fontsize=plot_style.COLUMN_FONT_SIZE)
        axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel(
        r"Offered normalized phase work ($\rho_f + \rho_d$)",
        fontsize=plot_style.COLUMN_FONT_SIZE)
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
                          label="Any repeat unstable, undrained, or failed"))
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


def plot_phase_surface(discovery: list[dict], heldout: list[dict],
                       out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=plot_style.COMPACT_FIGSIZE)
    for direction in plot_style.SERVICE_MIXES:
        selected = [row for row in discovery if row["direction"] == direction]
        if selected:
            axis.plot(
                [row["offered_prefill_rho_median"] for row in selected],
                [row["offered_decode_rho_median"] for row in selected],
                color=plot_style.SERVICE_MIX_COLORS[direction],
                linestyle=plot_style.SERVICE_MIX_LINESTYLES[direction],
                linewidth=1.5,
            )
            for row in selected:
                marker = plot_style.SERVICE_MIX_MARKERS[direction] \
                    if row["evidence_feasible"] else "x"
                axis.scatter(
                    row["offered_prefill_rho_median"],
                    row["offered_decode_rho_median"],
                    marker=marker, color=plot_style.SERVICE_MIX_COLORS[direction],
                    s=32, linewidths=1.2, zorder=3,
                )
        confirmed = [row for row in heldout if row["direction"] == direction]
        for row in confirmed:
            axis.scatter(
                row["offered_prefill_rho_median"],
                row["offered_decode_rho_median"],
                marker=plot_style.SERVICE_MIX_MARKERS[direction], s=72,
                facecolors="none", edgecolors=plot_style.SERVICE_MIX_COLORS[direction],
                linewidths=1.5, zorder=4,
            )
            if not row["evidence_feasible"]:
                axis.scatter(
                    row["offered_prefill_rho_median"],
                    row["offered_decode_rho_median"],
                    marker="x", color="#222222", s=34, linewidths=1.2,
                    zorder=5,
                )
    axis.set_xlabel(r"Prefill work $\rho_f$ (GPU-s/s)",
                    fontsize=plot_style.COLUMN_FONT_SIZE)
    axis.set_ylabel(r"Decode work $\rho_d$ (GPU-s/s)",
                    fontsize=plot_style.COLUMN_FONT_SIZE)
    axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
    axis.grid(alpha=.2)
    handles = [Line2D(
        [], [], color=plot_style.SERVICE_MIX_COLORS[direction],
        linestyle=plot_style.SERVICE_MIX_LINESTYLES[direction],
        marker=plot_style.SERVICE_MIX_MARKERS[direction],
        label=plot_style.SERVICE_MIX_NAMES[direction],
    ) for direction in plot_style.SERVICE_MIXES]
    handles.extend((
        Line2D([], [], color="#555555", marker="o", markerfacecolor="none",
               linestyle="none",
               label=plot_style.SERVICE_EVIDENCE_STAGE_NAMES["held_out"]),
        Line2D([], [], color="#222222", marker="x", linestyle="none",
               label="Any repeat missed evidence contract"),
    ))
    axis.legend(handles=handles, frameon=False, fontsize=7)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out.with_suffix(f".{suffix}"),
                       dpi=plot_style.SAVE_DPI, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--confirmed", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.confirmed) != bool(args.confirmation_plan):
        parser.error("--confirmed and --confirmation-plan must be supplied together")
    plan = campaign.read_plan(args.plan)
    rates = json.loads(args.normalization.read_text())
    campaign.validate_rates(rates)
    scout = json.loads(args.scout.read_text())
    campaign.validate_scout_evidence(plan, scout, scout["hardware"])
    if rates.get("sha256") != scout["normalization_sha256"]:
        raise RuntimeError("plot normalization differs from scout")
    confirmed = json.loads(args.confirmed.read_text()) if args.confirmed else None
    confirmation_plan = None
    if confirmed:
        confirmation_plan = campaign.read_plan(args.confirmation_plan)
        campaign.validate_confirmation_evidence(
            confirmed, confirmation_plan, plan, scout,
        )
    rows = aggregate(scout, plan, rates)
    write_csv(rows, args.out.with_suffix(".csv"))
    plot(rows, scout, confirmed, args.out)
    if confirmed and confirmation_plan:
        heldout = aggregate_confirmation(confirmed, confirmation_plan, rates)
        write_csv(heldout, args.out.with_name(
            f"{args.out.name}-heldout").with_suffix(".csv"))
        phase_rows = [*rows, *heldout]
        write_csv(phase_rows, args.out.with_name(
            f"{args.out.name}-phase").with_suffix(".csv"))
        plot_phase_surface(rows, heldout, args.out.with_name(
            f"{args.out.name}-phase"))


if __name__ == "__main__":
    main()
