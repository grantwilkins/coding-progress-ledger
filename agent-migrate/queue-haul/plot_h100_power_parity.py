"""Plot the frozen rational H100 power model against every measured cell."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style
from power_model_campaign import predict


STAGE_COUNTS = {"discovery": 90, "confirmation": 18, "idle": 3}
plot_style.apply()


def measured_rows(run_root: Path, fit: dict, cohort: str) -> list[dict]:
    cells = [json.loads(line) for line in (run_root / "cells.jsonl").read_text().splitlines()]
    if [row["sequence"] for row in cells] != list(range(len(cells))) or any(
            row["cached_prompt_tokens"] or
            (row["family"] != "idle" and not row["completed_requests"]) or
            row["power_samples"] / row["window_s"] < 5 for row in cells):
        raise ValueError("retrospective power cells fail accounting gates")
    rows = []
    gpu_uuid = json.loads((run_root / "metadata.json").read_text())["gpu"]["uuid"]
    for row in cells:
        predicted = predict(row, fit)
        rows.append({"sequence": row["sequence"], "stage": row["stage"],
                     "family": row["family"], "cohort": cohort,
                     "source_gpu_uuid": gpu_uuid,
                     "prompt_tokens": row["prompt_tokens"],
                     "output_tokens": row["output_tokens"],
                     "concurrency": row["concurrency"],
                     "realized_prefill_tps": row["realized_prefill_tps"],
                     "realized_decode_tps": row["realized_decode_tps"],
                     "predicted_power_w": predicted,
                     "measured_power_w": row["power_mean_w"],
                     "residual_w": row["power_mean_w"] - predicted})
    return rows


def load(run_root: Path, history_roots: tuple[Path, ...] | list[Path] = ()) \
        -> list[dict]:
    result = json.loads((run_root / "fit.json").read_text())
    if result["schema"] != "queue-haul-rational-power-fit-v1":
        raise ValueError("power parity requires the rational H100 fit")
    fit = result["fit"]
    cells = measured_rows(run_root, fit, "fit_campaign")
    counts = {stage: sum(row["stage"] == stage for row in cells) for stage in STAGE_COUNTS}
    if counts != STAGE_COUNTS or len(cells) != sum(STAGE_COUNTS.values()):
        raise ValueError("power parity requires the complete 111-cell campaign")
    primary_gpu = json.loads((run_root / "metadata.json").read_text())["gpu"]
    for root in history_roots:
        gpu = json.loads((root / "metadata.json").read_text())["gpu"]
        if (gpu["name"], gpu["power_limit_w"]) != (
                primary_gpu["name"], primary_gpu["power_limit_w"]):
            raise ValueError("retrospective power hardware is incompatible")
        cells += measured_rows(root, fit, "retrospective")
    return cells


def write(rows: list[dict], out: Path) -> None:
    values = [row[key] for row in rows
              for key in ("predicted_power_w", "measured_power_w")]
    margin = .04 * (max(values) - min(values))
    limits = min(values) - margin, max(values) + margin
    fig, axis = plt.subplots(figsize=plot_style.HALF_COLUMN_FIGSIZE)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.5,
              label="Ideal")
    for family in plot_style.POWER_FAMILY_NAMES:
        selected = [row for row in rows if row["family"] == family]
        if selected:
            axis.scatter([row["predicted_power_w"] for row in selected],
                         [row["measured_power_w"] for row in selected],
                         color=plot_style.POWER_FAMILY_COLORS[family],
                         marker=plot_style.POWER_FAMILY_MARKERS[family],
                         s=12, alpha=.58,
                         label=plot_style.POWER_FAMILY_NAMES[family])
    mean = statistics.fmean(row["measured_power_w"] for row in rows)
    residuals = [row["measured_power_w"] - row["predicted_power_w"]
                 for row in rows]
    r2 = 1 - sum(x*x for x in residuals) / sum(
        (row["measured_power_w"] - mean) ** 2 for row in rows)
    axis.set(xlabel="Predicted GPU\npower (W)", ylabel="Measured GPU power (W)",
             xlim=limits, ylim=limits)
    axis.text(.98, .03,
              f"MAE {statistics.fmean(map(abs, residuals)):.2f} W\n"
              f"$R^2$ {r2:.3f}",
              ha="right", va="bottom", transform=axis.transAxes,
              fontsize=plot_style.HALF_COLUMN_ANNOTATION_FONT_SIZE)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2, linewidth=.4)
    plot_style.half_column(axis)
    axis.legend(frameon=False, fontsize=plot_style.HALF_COLUMN_LEGEND_FONT_SIZE,
                loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2,
                columnspacing=.2, handlelength=1, handletextpad=.25,
                labelspacing=.2, borderaxespad=0)
    fig.tight_layout(pad=.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--history-run-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(load(args.run_root, args.history_run_root), args.out)


if __name__ == "__main__":
    main()
