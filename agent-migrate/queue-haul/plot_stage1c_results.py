from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS = Path(__file__).parent / "outputs"
CONDITIONS = ("admission", "mechanism_only")
LABELS = ("Admission-feasible", "Mechanism-only")
FLOATS = (
    "destination_load_budget_ell", "deadline_s", "target_w", "planned_source_drop_w",
    "source_baseline_w", "source_post_w", "measured_source_drop_w", "sink_baseline_w",
    "sink_post_w", "measured_sink_rise_w", "total_completion_s",
)
INTS = ("replicate", "sessions", "baseline_samples", "post_samples")
BOOLS = ("admission_feasible", "deadline_hit", "planned_hit", "power_hit")


def load_results(path: Path = OUTPUTS / "stage1c_results.csv") -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in FLOATS:
            row[key] = float(row[key])
        for key in INTS:
            row[key] = int(row[key])
        for key in BOOLS:
            row[key] = {"true": True, "false": False}[row[key].lower()]
        if abs(row["source_baseline_w"] - row["source_post_w"] - row["measured_source_drop_w"]) > 1e-9:
            raise ValueError("source power delta does not match baseline minus post")
        if abs(row["sink_post_w"] - row["sink_baseline_w"] - row["measured_sink_rise_w"]) > 1e-9:
            raise ValueError("sink power delta does not match post minus baseline")
    expected = {(condition, replicate) for condition in CONDITIONS for replicate in range(3)}
    if {(row["condition"], row["replicate"]) for row in rows} != expected:
        raise ValueError("stage 1c results require three paired replicates per condition")
    return rows


def _paired(ax, rows: list[dict], key: str, ylabel: str) -> None:
    values = {(row["condition"], row["replicate"]): row[key] for row in rows}
    for replicate in range(3):
        ax.plot((0, 1), [values[(condition, replicate)] for condition in CONDITIONS], color="0.75", zorder=1)
    for x, condition in enumerate(CONDITIONS):
        ys = [values[(condition, replicate)] for replicate in range(3)]
        ax.scatter([x] * 3, ys, s=36, zorder=2)
        ax.scatter(x, mean(ys), marker="D", color="black", s=35, zorder=3, label="mean" if x == 0 else None)
    ax.set(xticks=(0, 1), xticklabels=LABELS, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.25)


def _save(fig, base: Path) -> list[Path]:
    paths = [base.with_suffix(f".{suffix}") for suffix in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths


def write_plots(rows: list[dict], output_dir: Path = OUTPUTS) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _paired(axes[0], rows, "measured_source_drop_w", "Source power drop (W)")
    _paired(axes[1], rows, "measured_sink_rise_w", "Sink power rise (W)")
    for x, condition in enumerate(CONDITIONS):
        planned = mean(row["planned_source_drop_w"] for row in rows if row["condition"] == condition)
        axes[0].scatter(x, planned, marker="_", s=350, linewidth=2, color="tab:orange", label="planned" if x == 0 else None)
    axes[0].axhline(rows[0]["target_w"], color="tab:red", linestyle="--", label="target")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage 1c paired power validation")
    fig.tight_layout()
    paths = _save(fig, output_dir / "stage1c_power_change")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, node in zip(axes, ("source", "sink")):
        for condition, label, color in zip(CONDITIONS, LABELS, ("tab:blue", "tab:orange")):
            for row in (row for row in rows if row["condition"] == condition):
                ax.plot((0, 1), (row[f"{node}_baseline_w"], row[f"{node}_post_w"]),
                        marker="o", color=color, alpha=0.75, label=label if row["replicate"] == 0 else None)
        ax.set(xticks=(0, 1), xticklabels=("Baseline", "Post"), ylabel="Power (W)", title=f"{node.title()} GPU")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Most recent valid Stage 1c runs: power before and after")
    fig.tight_layout()
    paths += _save(fig, output_dir / "stage1c_power_levels")

    fig, ax = plt.subplots(figsize=(5, 4))
    _paired(ax, rows, "total_completion_s", "Total completion time (s, log scale)")
    ax.axhline(rows[0]["deadline_s"], color="tab:red", linestyle="--", label="deadline")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return paths + _save(fig, output_dir / "stage1c_completion")


if __name__ == "__main__":
    write_plots(load_results())
