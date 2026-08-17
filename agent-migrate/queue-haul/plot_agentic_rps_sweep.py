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
    ("p90_ttft_s", "P90 TTFT (s)", 1),
    ("p90_tpot_s", "P90 TPOT (ms)", 1000),
)
SLOS = {"p90_ttft_s": 1.0, "p90_tpot_s": .05}


def plot_summary(summary: dict, output: Path, h100: dict | None = None) -> None:
    result = summary["models"][MODEL]
    curves = {"a100": result["curve"]}
    if h100 is not None:
        if h100.get("schema") != SCHEMA or h100.get("stage") != "reduced" or \
                summary.get("hardware") != "a100" or \
                h100.get("hardware") != "h100" or \
                h100.get("request_shape") != summary.get("request_shape"):
            raise ValueError("H100 curve does not match the agentic request shape")
        h100_result = h100["models"][MODEL]
        if h100_result["slo"] != result["slo"]:
            raise ValueError("A100 and H100 SLOs do not match")
        curves["h100"] = h100_result["curve"]
    figure, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    for axis, (field, ylabel, scale) in zip(axes, METRICS):
        for hardware, rows in curves.items():
            rows = sorted(rows, key=lambda row: row["offered_rps"])
            axis.plot(
                [row["offered_rps"] for row in rows],
                [row[f"{field}_median"] * scale for row in rows],
                label=(plot_style.AGENTIC_WORKLOAD_NAME if h100 is None else
                       f"Agent - {plot_style.AGENTIC_HARDWARE_NAMES[hardware]}"),
                color=plot_style.AGENTIC_HARDWARE_COLORS[hardware],
                marker=plot_style.AGENTIC_HARDWARE_MARKERS[hardware],
                linewidth=1.5,
            )
        axis.axhline(
            SLOS[field] * scale, color="black", linestyle=":",
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
    parser.add_argument("--h100-summary", type=Path)
    args = parser.parse_args(argv)
    h100 = json.loads(args.h100_summary.read_text()) \
        if args.h100_summary else None
    plot(json.loads(args.summary.read_text()), args.output_dir, h100)


if __name__ == "__main__":
    main()
