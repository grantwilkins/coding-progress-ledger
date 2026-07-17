"""Plot the session-scaling solver comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"node_aware": "#4C78A8", "node_drain": "#F58518", "lp": "#54A24B"}
LABELS = {"node_aware": "Node aware", "node_drain": "Drain nodes", "lp": "LP"}
PAIRED_FIELDS = (
    "source_instances", "source_nodes", "bandwidth_gbps_per_node", "deadline_s", "end_s",
    "target_fraction_of_removable_power", "requested_source_drop_w",
)


def ratios(row: dict) -> tuple[float, float, float]:
    selected = 100 * row["planned_moves"] / row["sessions"]
    completed = 100 * row["moves_completed_by_deadline"] / row["planned_moves"]
    achieved = 100 * row["modeled_source_drop_at_deadline_w"] / row["requested_source_drop_w"]
    return selected, completed, achieved


def plot_title(row: dict) -> str:
    return (
        f"Coding, {row['bandwidth_gbps_per_node']:g} Gbps/node, "
        f"{row['deadline_s'] / 60:g} min deadline, "
        f"{100 * row['target_fraction_of_removable_power']:g}% awake-state power reduction"
    )


def read_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("scaling results are empty")
    numeric = set(rows[0]) - {"solver", "plan_feasible"}
    converted = [{key: float(value) if key in numeric else value for key, value in row.items()}
                 for row in rows]
    solvers = sorted({row["solver"] for row in converted})
    sessions = sorted({row["sessions"] for row in converted})
    indexed = {(row["solver"], row["sessions"]): row for row in converted}
    if len(indexed) != len(converted) or set(indexed) != set(
        (solver, count) for solver in solvers for count in sessions
    ):
        raise ValueError("every solver requires exactly one row per session count")
    for count in sessions:
        reference = indexed[solvers[0], count]
        if any(not math.isclose(indexed[solver, count][field], reference[field],
                                rel_tol=1e-12, abs_tol=1e-9)
               for solver in solvers[1:] for field in PAIRED_FIELDS):
            raise ValueError(f"solver settings differ at {count:g} sessions")
    return converted


def plot(rows: list[dict], output: Path) -> None:
    solvers = sorted({row["solver"] for row in rows}, key=lambda value: tuple(COLORS).index(value))
    series = {
        solver: sorted((row for row in rows if row["solver"] == solver),
                       key=lambda row: row["sessions"])
        for solver in solvers
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for solver, selected in series.items():
        x = [row["sessions"] for row in selected]
        values = [ratios(row) for row in selected]
        style = dict(marker="o", color=COLORS[solver], label=LABELS[solver])
        axes[0, 0].plot(x, [value[0] for value in values], **style)
        axes[0, 1].plot(x, [value[1] for value in values], **style)
        axes[0, 2].plot(x, [value[2] for value in values], **style)
        axes[1, 0].plot(
            x, [row["modeled_source_drop_at_deadline_w"] / 1000 for row in selected], **style
        )
        axes[1, 1].plot(x, [row["makespan_s"] for row in selected], **style)
        axes[1, 2].plot(x, [row["plan_s"] for row in selected], **style)
    reference = next(iter(series.values()))
    axes[1, 0].plot(
        [row["sessions"] for row in reference],
        [row["requested_source_drop_w"] / 1000 for row in reference],
        "k--", label="Target",
    )
    axes[0, 2].axhline(100, color="black", linestyle="--", label="Target")
    axes[1, 1].axhline(reference[0]["deadline_s"], color="black", linestyle="--",
                       label="Deadline")
    titles = (
        "Sessions selected", "Selected moves completed by deadline",
        "Requested power reduction achieved", "Power reduction at deadline",
        "Last completed migration", "Planning time",
    )
    ylabels = ("Percent", "Percent", "Percent", "GPU power (kW)", "Seconds", "Seconds")
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set(title=title, xlabel="Sessions", ylabel=ylabel, xscale="log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1, 2].set_yscale("log")
    fig.suptitle(plot_title(reference[0]))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent / "outputs/scaling_1_to_100k_20260716/scaling_results.csv",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.input)
    plot(rows, args.output or args.input.with_name("scaling_summary"))


if __name__ == "__main__":
    main()
