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


def aggregate_transition(result: dict, baseline_work: float) -> list[dict]:
    """Reduce the live transition to one conservative point per mix.

    Each metric is the worse of the incumbent and newly admitted cohort for a
    restart block.  The plotted range is the observed minimum--maximum across
    the three blocks, not a confidence interval.
    """
    if result.get("campaign_pass") is not True \
            or result.get("planner_usable") is not False:
        raise RuntimeError("transition evidence is not a complete passing run")
    rows = []
    for direction in plot_style.SERVICE_MIXES:
        selected = [row for row in result.get("rows", ())
                    if row.get("direction") == direction]
        if len(selected) != 3 or {row.get("block") for row in selected} \
                != {6, 7, 8}:
            raise RuntimeError("transition evidence does not cover three blocks")
        total_work = {row["offered_coordinates"]["offered_rho"]
                      for row in selected}
        if len(total_work) != 1:
            raise RuntimeError("transition mix changed work across blocks")
        values = {}
        for metric in ("p90_ttft_s", "p90_mean_tpot_s"):
            block_values = [max(
                row["windows"]["post_admission"][metric],
                row["new_cohort"][metric],
            ) for row in selected]
            values[metric] = block_values
        rows.append({
            "stage": "transition", "direction": direction,
            "blocks": len(selected),
            "total_work": next(iter(total_work)),
            "added_work": next(iter(total_work)) - baseline_work,
            **{f"{metric}_median": statistics.median(metric_values)
               for metric, metric_values in values.items()},
            **{f"{metric}_minimum": min(metric_values)
               for metric, metric_values in values.items()},
            **{f"{metric}_maximum": max(metric_values)
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


def workload_panels(rows: list[dict], targets: dict) -> list[dict]:
    """Return raw latency slices along the prefill-heavy workload ray."""
    trajectory = sorted(
        (row for row in rows if row["direction"] == "prefill_heavy"),
        key=lambda row: row["target_rho"],
    )
    panels = []
    for metric, x_field, ylabel, xlabel in (
        ("p90_ttft_s", "offered_prefill_rho_median", "P90 TTFT (s)",
         r"Prefill work, $\rho_p$ (GPU-s/s)"),
        ("p90_mean_tpot_s", "offered_decode_rho_median",
         "P90 mean TPOT (s)", r"Decode work, $\rho_d$ (GPU-s/s)"),
    ):
        misses = [row for row in trajectory
                  if row[f"{metric}_median"] > targets[metric]]
        if not misses:
            raise RuntimeError(f"{metric} has no measured SLO violation")
        panels.append({
            "metric": metric, "ylabel": ylabel, "xlabel": xlabel,
            "x": [row[x_field] for row in trajectory],
            "y": [row[f"{metric}_median"] for row in trajectory],
            "first_miss": misses[0], "x_field": x_field,
        })
    return panels


def plot(rows: list[dict], scout: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    direction = "prefill_heavy"
    panels = workload_panels(rows, scout["targets"])
    figure, axes = plt.subplots(
        2, 1, figsize=(plot_style.COMPACT_FIGSIZE[0], 5.2),
    )
    for axis, panel in zip(axes, panels):
        metric = panel["metric"]
        axis.plot(
            panel["x"], panel["y"],
            color=plot_style.SERVICE_MIX_COLORS[direction],
            linestyle=plot_style.SERVICE_MIX_LINESTYLES[direction],
            marker=plot_style.SERVICE_MIX_MARKERS[direction], markersize=5,
            linewidth=1.8,
        )
        target = scout["targets"][metric]
        axis.axhline(target, color="#222222", linestyle="--", linewidth=1.4)
        if metric == "p90_ttft_s":
            axis.set_yscale("log")
        miss = panel["first_miss"]
        miss_x = miss[panel["x_field"]]
        axis.scatter(miss_x, miss[f"{metric}_median"], marker="x",
                     color="#222222", s=48, linewidths=1.4, zorder=4)
        axis.vlines(
            miss_x, axis.get_ylim()[0], target,
            color=plot_style.SERVICE_MIX_COLORS[direction],
            linestyle="--", linewidth=1.2,
        )
        offset, alignment = ((-5, 15), "right") \
            if metric == "p90_ttft_s" else ((5, 15), "left")
        axis.annotate(
            rf"$\rho={miss['target_rho']:.2f}$",
            xy=(miss_x, 0), xycoords=("data", "axes fraction"),
            xytext=offset, textcoords="offset points", ha=alignment,
            fontsize=plot_style.COLUMN_FONT_SIZE,
            arrowprops={"arrowstyle": "->", "color": "#222222", "lw": .8},
        )
        axis.set_ylabel(panel["ylabel"], fontsize=plot_style.COLUMN_FONT_SIZE)
        axis.set_xlabel(panel["xlabel"], fontsize=plot_style.COLUMN_FONT_SIZE)
        axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
        axis.grid(alpha=.2)
    figure.legend(handles=(
        Line2D([], [], color=plot_style.SERVICE_MIX_COLORS[direction],
               linestyle=plot_style.SERVICE_MIX_LINESTYLES[direction],
               marker=plot_style.SERVICE_MIX_MARKERS[direction],
               label=plot_style.SERVICE_MIX_NAMES[direction]),
        Line2D([], [], color="#222222", linestyle="--", label="SLO"),
        Line2D([], [], color="#222222", marker="x", linestyle="none",
               label="First measured violation"),
    ), frameon=False, fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE, ncol=2,
        loc="upper center", bbox_to_anchor=(.5, .995))
    figure.tight_layout(rect=(0, 0, 1, .88))
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


def plot_sustainability(discovery: list[dict], transition: list[dict],
                        scout: dict, out: Path) -> None:
    """Retain the historical output name for the simple measured slices."""
    if not transition:
        raise RuntimeError("transition evidence is required")
    plot(discovery, scout, out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--confirmed", type=Path)
    parser.add_argument("--confirmation-plan", type=Path)
    parser.add_argument("--transition", type=Path)
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
    heldout = []
    if confirmed and confirmation_plan:
        heldout = aggregate_confirmation(confirmed, confirmation_plan, rates)
        write_csv(heldout, args.out.with_name(
            f"{args.out.name}-heldout").with_suffix(".csv"))
        phase_rows = [*rows, *heldout]
        write_csv(phase_rows, args.out.with_name(
            f"{args.out.name}-phase").with_suffix(".csv"))
        plot_phase_surface(rows, heldout, args.out.with_name(
            f"{args.out.name}-phase"))
    if args.transition:
        transition_result = json.loads(args.transition.read_text())
        baseline_work = statistics.median(
            row["measured_rho_median"] for row in rows
            if row["target_rho"] == campaign.BASE_RHO
        )
        transition = aggregate_transition(transition_result, baseline_work)
        write_csv(transition, args.out.with_name(
            f"{args.out.name}-sustainability-transition").with_suffix(".csv"))
        plot_sustainability(
            rows, transition, scout,
            args.out.with_name(f"{args.out.name}-sustainability"),
        )
    plot(rows, scout, args.out)


if __name__ == "__main__":
    main()
