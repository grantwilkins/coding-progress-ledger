"""Plot the compact GPT-OSS-20B agentic RPS sweep figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import plot_style


plot_style.apply()

MODEL = "openai/gpt-oss-20b"
SCHEMA = "queue-haul-agentic-rps-sweep-v3"
METRICS = (
    ("p90_ttft_s", "P90 TTFT (s)"),
    ("p90_tpot_s", "P90 TPOT (s)"),
)


def plot_summary(summary: dict, output: Path, h100: dict | None = None) -> None:
    result = summary["models"][MODEL]
    curves = {"a100": result["curve"]}
    if h100 is not None:
        contract = h100["contract"]
        if contract["hardware"] != "h100" or \
                (contract["input_tokens"], contract["output_tokens"],
                 contract["requests_per_point"]) != (3920, 1024, 32):
            raise ValueError("H100 curve does not match the agentic request shape")
        curves["h100"] = [row for row in h100["points"]
                           if row["model"] == MODEL]
    figure, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    for axis, (field, ylabel) in zip(axes, METRICS):
        for hardware, rows in curves.items():
            rows = sorted(rows, key=lambda row: row["offered_rps"])
            suffix = "_median" if hardware == "a100" else ""
            axis.plot(
                [row["offered_rps"] for row in rows],
                [row[f"{field}{suffix}"] for row in rows],
                label=(plot_style.AGENTIC_WORKLOAD_NAME if h100 is None else
                       f"{plot_style.AGENTIC_WORKLOAD_NAME} · "
                       f"{plot_style.AGENTIC_HARDWARE_NAMES[hardware]}"),
                color=plot_style.AGENTIC_HARDWARE_COLORS[hardware],
                marker=plot_style.AGENTIC_HARDWARE_MARKERS[hardware],
                linewidth=1.5,
            )
        axis.axhline(
            result["slo"][field], color="black", linestyle=":",
            linewidth=1.2, label="SLO",
        )
        axis.set_ylabel(ylabel, fontsize=plot_style.COLUMN_FONT_SIZE)
        axis.tick_params(labelsize=plot_style.COLUMN_FONT_SIZE)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Rate (req/s)", fontsize=plot_style.COLUMN_FONT_SIZE)
    axes[-1].set_xlim(0, max(row["offered_rps"] for rows in curves.values()
                             for row in rows) + .25)
    axes[0].legend(frameon=False, ncol=1, loc="upper left",
                   fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE)
    figure.subplots_adjust(left=.19, right=.97, bottom=.13, top=.97, hspace=.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(figure)


def plot(summary: dict, output_dir: Path, h100: dict | None = None) -> Path:
    if summary.get("schema") != SCHEMA or summary.get("stage") != "reduced":
        raise ValueError("agentic RPS summary is not reduced evidence")
    output = output_dir / "agentic-rps-sweep"
    plot_summary(summary, output, h100)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--h100-curve", type=Path)
    args = parser.parse_args(argv)
    h100 = json.loads(args.h100_curve.read_text()) if args.h100_curve else None
    plot(json.loads(args.summary.read_text()), args.output_dir, h100)


if __name__ == "__main__":
    main()
