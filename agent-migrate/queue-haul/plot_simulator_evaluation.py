"""Plot achieved power reduction, route-switch time, and request wait."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planner import SOLVERS, plan, source_power
from power_drain_experiment import (
    DEFAULT_MODEL,
    DEFAULT_WORKLOADS,
    ExperimentRun,
    _summary,
    build_scenario,
)
from profiles import ModelProfile, WorkloadProfile
from simulate import execute


SESSIONS = 50
DEADLINE_S = 10
END_S = 15
TARGET_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
COLORS = {
    "random": "#9B51E0",
    "load_only": "#F2994A",
    "node_aware": "#2F80ED",
    "node_drain": "#176B52",
}
LABELS = {
    "random": "Random",
    "load_only": "Load only",
    "node_aware": "Node aware",
    "node_drain": "Drain nodes",
}


def evaluation_rows() -> list[dict]:
    profile = ModelProfile.load(DEFAULT_MODEL)
    rows = []
    for path in DEFAULT_WORKLOADS:
        full = WorkloadProfile.load(path)
        # TODO(workload-range): use full held-out traces once they stay within the measured context range.
        workload = replace(
            full,
            profile_id=f"{full.profile_id}-lowest-context",
            records=full.records[:1],
        )
        probe, control_routes = build_scenario(
            workload, profile, SESSIONS, 3, 0, DEADLINE_S, END_S
        )
        initial = source_power(probe, profile)
        minimum = source_power(
            probe, profile, (session.session_id for session in probe.sessions)
        )
        control_scenario = replace(probe, power_limit_w=initial)
        control_plan = plan(
            control_scenario, profile, control_routes, "load_only", seed=3
        )
        control_result = execute(control_scenario, profile, control_plan.moves)
        control_wait = _summary(
            ExperimentRun(
                f"{workload.profile_id}:control",
                workload.profile_id,
                control_scenario,
                control_plan,
                "central",
                control_result,
            )
        )["p95_request_wait_s"]
        for fraction in TARGET_FRACTIONS:
            limit = initial - fraction * (initial - minimum)
            scenario, routes = build_scenario(
                workload, profile, SESSIONS, 3, limit, DEADLINE_S, END_S
            )
            for solver in SOLVERS:
                planned = plan(scenario, profile, routes, solver, seed=3)
                result = execute(scenario, profile, planned.moves)
                run = ExperimentRun(
                    f"{workload.profile_id}:{fraction}:{solver}",
                    workload.profile_id,
                    scenario,
                    planned,
                    "central",
                    result,
                )
                rows.append(
                    {
                        **_summary(run),
                        "job_type": workload.records[0].job_type,
                        "requested_fraction_of_available_drop": fraction,
                        "migration_completion_s": (
                            result.makespan_s
                            if result.completed_sessions == len(planned.moves)
                            else scenario.end_s
                        ),
                        "control_p95_request_wait_s": control_wait,
                    }
                )
    return rows


def _plot(rows: list[dict], out: Path) -> None:
    jobs = [
        WorkloadProfile.load(path).records[0].job_type for path in DEFAULT_WORKLOADS
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for column, job in enumerate(jobs):
        selected = [row for row in rows if row["job_type"] == job]
        maximum = max(row["requested_source_drop_w"] for row in selected) / 1000
        axes[0, column].plot([0, maximum], [0, maximum], "--", color="#98A2B3")
        for solver in SOLVERS:
            series = sorted(
                (row for row in selected if row["solver"] == solver),
                key=lambda row: row["requested_source_drop_w"],
            )
            requested = [row["requested_source_drop_w"] / 1000 for row in series]
            achieved = [
                row["modeled_source_drop_at_deadline_w"] / 1000 for row in series
            ]
            color = COLORS[solver]
            axes[0, column].plot(
                requested,
                achieved,
                "o-",
                color=color,
                label=LABELS[solver],
            )
            axes[1, column].plot(
                achieved,
                [row["migration_completion_s"] for row in series],
                "o-",
                color=color,
            )
            axes[2, column].plot(
                achieved,
                [row["p95_request_wait_s"] for row in series],
                "o-",
                color=color,
            )
            values = (
                (requested, achieved, "power_met"),
                (
                    achieved,
                    [row["migration_completion_s"] for row in series],
                    "moves_committed_by_deadline",
                ),
                (
                    achieved,
                    [row["p95_request_wait_s"] for row in series],
                    "requests_started_by_deadline",
                ),
            )
            for row_index, (x, y, field) in enumerate(values):
                missed = [index for index, row in enumerate(series) if not row[field]]
                axes[row_index, column].scatter(
                    [x[index] for index in missed],
                    [y[index] for index in missed],
                    marker="x",
                    s=70,
                    linewidth=2,
                    color=color,
                )
        axes[0, column].set_title(job.replace("_", " ").title())
        axes[0, column].set_ylabel(
            "Power reduction by 10 s (kW)" if column == 0 else ""
        )
        axes[1, column].set_ylabel("Last route switch (s)" if column == 0 else "")
        axes[2, column].set_ylabel(
            "95th-percentile request wait (s)" if column == 0 else ""
        )
        axes[1, column].axhline(DEADLINE_S, linestyle="--", color="#98A2B3")
        axes[2, column].axhline(
            selected[0]["control_p95_request_wait_s"],
            linestyle="--",
            color="#98A2B3",
        )
        axes[1, column].set_xlabel("Power reduction by 10 s (kW)")
        axes[2, column].set_xlabel("Power reduction by 10 s (kW)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_xlabel("Requested source-power reduction (kW)")
    for axis in axes[0, 1:]:
        axis.set_xlabel("Requested source-power reduction (kW)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Power targets and session disruption in the new simulator",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.93,
        "50 sessions per workload at the lowest measured context · current estimated GPT-OSS-20B/A100 profile · 1 Gbps · 10 s deadline",
        ha="center",
        color="#667085",
    )
    fig.text(
        0.01,
        0.01,
        "Bottom dashed line: same requests without migration",
        color="#667085",
    )
    fig.text(
        0.99,
        0.01,
        "× = that row's deadline missed",
        ha="right",
        color="#667085",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    for extension in ("png", "pdf"):
        fig.savefig(
            out / f"simulator_evaluation.{extension}",
            dpi=180,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def write(out: Path) -> None:
    rows = evaluation_rows()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "simulator_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    write(parser.parse_args().out)


if __name__ == "__main__":
    main()
