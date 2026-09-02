"""Plot the compact GPT-OSS-20B agentic RPS sweep figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import plot_style


plot_style.apply()

MODEL = "openai/gpt-oss-20b"
SCHEMA = "queue-haul-agentic-rps-sweep-v3"
SLO_SCHEMA = "queue-haul-agentic-rps-sweep-v4"
QUICK_SCHEMA = "queue-haul-quick-slo-sweep-v4"
ERROR_BAR_SCHEMAS = {SLO_SCHEMA, QUICK_SCHEMA}
METRICS = (
    ("p90_ttft_s", "P90 TTFT (s)", 1),
    ("p90_tpot_s", "P90 TPOT (ms)", 1000),
)
SLOS = {"p90_ttft_s": 1.0, "p90_tpot_s": .05}


def plot_summary(summary: dict, output: Path, h100: dict | None = None) -> None:
    result = summary["models"][MODEL]
    schema = summary["schema"]
    hardware = summary.get("hardware", "a100")
    curves = {hardware: result["curve"]}
    slos = (SLOS if schema == SCHEMA else
            {field: result["slo"][field] for field, *_ in METRICS})
    if h100 is not None:
        if h100.get("schema") != schema or h100.get("stage") != "reduced" or \
                summary.get("hardware") != "a100" or \
                h100.get("hardware") != "h100" or \
                h100.get("request_shape") != summary.get("request_shape") or \
                (schema in ERROR_BAR_SCHEMAS and (
                    h100.get("comparison_sha256") !=
                    summary.get("comparison_sha256") or
                    h100.get("shared_runtime_sha256") !=
                    summary.get("shared_runtime_sha256") or
                    h100.get("launch_git_sha") !=
                    summary.get("launch_git_sha")
                )):
            raise ValueError("H100 curve does not match the agentic request shape")
        h100_result = h100["models"][MODEL]
        if h100_result["slo"] != result["slo"]:
            raise ValueError("A100 and H100 SLOs do not match")
        curves["h100"] = h100_result["curve"]
    figure, axes = plt.subplots(
        1, 2, figsize=(5.2, 2.5) if h100 else (3.35, 2.1), sharex=True,
    )
    font_size = (plot_style.COLUMN_FONT_SIZE if h100 else
                 plot_style.HALF_COLUMN_FONT_SIZE)
    for axis, (field, ylabel, scale) in zip(axes, METRICS):
        failure_labeled = False
        violation_labeled = False
        for hardware, rows in curves.items():
            rows = sorted(rows, key=lambda row: row["offered_rps"])
            label = (plot_style.AGENTIC_WORKLOAD_NAME if h100 is None else
                     f"Agent - {plot_style.AGENTIC_HARDWARE_NAMES[hardware]}")
            style = {
                "label": label,
                "color": plot_style.AGENTIC_HARDWARE_COLORS[hardware],
                "linestyle": plot_style.AGENTIC_HARDWARE_LINESTYLES[hardware],
                "linewidth": 1.1,
            }
            if schema in ERROR_BAR_SCHEMAS:
                points = [point for row in rows for point in row["points"]
                          if point["status"] == "numeric"
                          and point[field] is not None]
                axis.scatter(
                    [point["realized_rps"] for point in points],
                    [point[field] * scale for point in points],
                    color=style["color"], alpha=.22, s=8, linewidths=0,
                    zorder=1,
                )
                usable = [row for row in rows
                          if row[f"{field}_median"] is not None
                          and row[f"{field}_ci_low"] is not None
                          and row[f"{field}_ci_high"] is not None]
                medians = [row[f"{field}_median"] * scale for row in usable]
                axis.errorbar(
                    [row["offered_rps"] for row in usable], medians,
                    yerr=[
                        [median - row[f"{field}_ci_low"] * scale
                         for row, median in zip(usable, medians)],
                        [row[f"{field}_ci_high"] * scale - median
                         for row, median in zip(usable, medians)],
                    ], marker="o", markevery=[
                        index for index, row in enumerate(usable)
                        if row[f"{field}_median"] <= slos[field]
                    ], capsize=2, elinewidth=.8, zorder=3, **style,
                )
                violations = [row for row in usable
                              if row[f"{field}_median"] > slos[field]]
                if violations:
                    axis.scatter(
                        [row["offered_rps"] for row in violations],
                        [row[f"{field}_median"] * scale for row in violations],
                        color=style["color"],
                        marker=plot_style.SLO_VIOLATION_MARKER, s=18, zorder=4,
                        label=(plot_style.SLO_VIOLATION_NAME
                               if not violation_labeled else "_nolegend_"),
                    )
                    violation_labeled = True
                failures = [point for row in rows for point in row["points"]
                            if point["status"] == "service_failure"]
                if failures:
                    axis.scatter(
                        [point["realized_rps"] if point["realized_rps"] is not None
                         else row["offered_rps"]
                         for row in rows for point in row["points"]
                         if point["status"] == "service_failure"],
                        [1.02] * len(failures),
                        transform=axis.get_xaxis_transform(),
                        color=style["color"], marker="x", s=14, zorder=4,
                        clip_on=False,
                        label=("Censored service failure"
                               if not failure_labeled else "_nolegend_"),
                    )
                    failure_labeled = True
            else:
                axis.plot(
                    [row["offered_rps"] for row in rows],
                    [row[f"{field}_median"] * scale for row in rows],
                    marker=plot_style.AGENTIC_HARDWARE_MARKERS[hardware],
                    **style,
                )
        axis.axhline(
            slos[field] * scale, color=plot_style.SLO_COLOR,
            linestyle=plot_style.SLO_LINESTYLE, linewidth=1.7,
            label=plot_style.SLO_NAME, zorder=5,
        )
        axis.set_ylabel(ylabel, fontsize=font_size)
        axis.set_xlabel("Rate (req/s)", fontsize=font_size)
        axis.tick_params(labelsize=font_size)
        axis.grid(alpha=.2)
    if schema in ERROR_BAR_SCHEMAS:
        x_values = [value for rows in curves.values() for row in rows
                    for value in (
                        row["offered_rps"], row["realized_rps_median"],
                        *(point["realized_rps"] for point in row["points"]),
                    ) if value is not None and value > 0]
        minimum, maximum = min(x_values), max(x_values)
        for axis in axes:
            axis.set_xscale("log", base=2)
            axis.set_xlim(minimum / 1.15, maximum * 1.15)
            axis.xaxis.set_major_locator(ticker.LogLocator(base=2))
            axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            axis.xaxis.set_minor_locator(ticker.NullLocator())
            axis.set_ylim(bottom=0)
    else:
        for axis in axes:
            axis.set_xlim(0, 8.25)
            axis.set_xticks((0, 2, 4, 6, 8))
        axes[1].set_ylim(0, 52)
        axes[1].set_yticks((0, 10, 20, 30, 40, 50))
    legend = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    handles, labels = zip(*sorted(
        ((handle, label) for label, handle in legend.items()),
        key=lambda pair: (pair[1] == "Censored service failure",
                          pair[1] == plot_style.SLO_NAME),
    ))
    censored = "Censored service failure" in labels
    figure.legend(handles, labels, frameon=False,
                  ncol=2 if censored else len(handles),
                  loc="upper center",
                  bbox_to_anchor=(.5, .99 if censored else .94),
                  fontsize=(plot_style.COLUMN_LEGEND_FONT_SIZE if h100 else
                            plot_style.HALF_COLUMN_LEGEND_FONT_SIZE))
    figure.tight_layout(rect=(0, 0, 1, .72 if censored else .82))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                       bbox_inches="tight")
    plt.close(figure)


def plot(summary: dict, output_dir: Path, h100: dict | None = None) -> Path:
    if summary.get("schema") not in {SCHEMA, *ERROR_BAR_SCHEMAS} \
            or summary.get("stage") != "reduced":
        raise ValueError("agentic RPS summary is not reduced evidence")
    if summary["schema"] in ERROR_BAR_SCHEMAS and any(
            not isinstance(summary.get(key), str) for key in (
                "comparison_sha256", "shared_runtime_sha256", "launch_git_sha",
            )):
        raise ValueError("agentic SLO summary lacks runtime provenance")
    output = output_dir / "agentic-rps-sweep"
    plot_summary(summary, output, h100)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--h100-summary", type=Path)
    args = parser.parse_args(argv)
    h100 = json.loads(args.h100_summary.read_text()) \
        if args.h100_summary else None
    plot(json.loads(args.summary.read_text()), args.output_dir, h100)


if __name__ == "__main__":
    main()
