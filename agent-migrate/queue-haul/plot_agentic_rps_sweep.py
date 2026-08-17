"""Plot the compact GPT-OSS-20B agentic RPS sweep figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import plot_style


plot_style.apply()

MODEL = "openai/gpt-oss-20b"
SCHEMA = "queue-haul-agentic-rps-sweep-v2"
METRICS = (
    ("p90_ttft_s", "P90 TTFT (s)"),
    ("p90_tpot_s", "P90 TPOT (s)"),
)


def plot_summary(summary: dict, output: Path) -> None:
    result = summary["models"][MODEL]
    rows = sorted(result["curve"], key=lambda row: row["offered_rps"])
    figure, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    for axis, (field, ylabel) in zip(axes, METRICS):
        usable = [row for row in rows if row.get(f"{field}_median") is not None]
        axis.plot(
            [row["offered_rps"] for row in usable],
            [row[f"{field}_median"] for row in usable],
            label=plot_style.AGENTIC_WORKLOAD_NAME,
            color=plot_style.AGENTIC_WORKLOAD_COLOR,
            marker=plot_style.AGENTIC_WORKLOAD_MARKER,
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
    axes[-1].set_xlim(0, max(row["offered_rps"] for row in rows) + .25)
    axes[0].legend(frameon=False, ncol=2, loc="upper left",
                   fontsize=plot_style.COLUMN_LEGEND_FONT_SIZE)
    figure.subplots_adjust(left=.19, right=.97, bottom=.13, top=.97, hspace=.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(figure)


def plot(summary: dict, output_dir: Path) -> Path:
    if summary.get("schema") != SCHEMA or summary.get("stage") != "reduced":
        raise ValueError("agentic RPS summary is not reduced evidence")
    output = output_dir / "agentic-rps-sweep"
    plot_summary(summary, output)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    plot(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
