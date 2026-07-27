"""Canonical Q1-Q9 result and plot registry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from evaluation_config import evaluation_manifest


@dataclass(frozen=True)
class PlotSpec:
    question: str
    name: str
    filename: str
    required_fields: tuple[str, ...]


PLOTS = (
    PlotSpec("Q1", "group power", "q1_group_power.pdf",
             ("predicted_shed_w", "measured_shed_w", "safety_margin_w")),
    PlotSpec("Q1", "power window", "q1_power_window.pdf",
             ("load", "power_w", "averaging_window_s")),
    PlotSpec("Q2", "action phase", "q2_action_phase.pdf",
             ("bandwidth_gbps", "context_tokens", "method", "handoff_s")),
    PlotSpec("Q2", "handoff breakdown", "q2_handoff_breakdown.pdf",
             ("method", "route_s", "reconstruction_s", "catch_up_s", "pause_s")),
    PlotSpec("Q3", "action mix", "q3_action_mix.pdf",
             ("route_headroom", "service_headroom", "replay_fraction", "shed_w")),
    PlotSpec("Q4", "shed target", "q4_achieved_shed.pdf",
             ("requested_shed_w", "achieved_shed_w", "unmet_shed_w")),
    PlotSpec("Q4", "requirement frontier", "q4_requirement_frontier.pdf",
             ("shed_w", "route_bytes", "service_work", "kv_blocks",
              "service_debt_replica_s", "required_recovery_s")),
    PlotSpec("Q4", "binding map", "q4_binding_map.pdf",
             ("route_headroom", "service_headroom", "binding_resources", "shed_w")),
    PlotSpec("Q5", "makespan validation", "q5_makespan.pdf",
             ("predicted_makespan_s", "realized_makespan_s")),
    PlotSpec("Q5", "drain timeline", "q5_drain_timeline.pdf",
             ("time_s", "source_power_w", "route_bytes_per_s", "queued_work")),
    PlotSpec("Q6", "planner quality", "q6_planner_quality.pdf",
             ("planner", "shed_w", "upper_bound_w", "solve_s")),
    PlotSpec("Q6", "planner scaling", "q6_planner_scaling.pdf",
             ("sessions", "candidates", "planner", "solve_s", "memory_bytes")),
    PlotSpec("Q7", "pool value", "q7_pool_value.pdf",
             ("pools", "budget_policy", "shed_w", "binding_resources")),
    PlotSpec("Q7", "pool diversity", "q7_pool_diversity.pdf",
             ("diversity_case", "compatibility", "shed_w")),
    PlotSpec("Q8", "resource improvement", "q8_resource_sensitivity.pdf",
             ("resource", "multiplier", "shed_w", "binding_resources")),
    PlotSpec("Q8", "policy changes", "q8_policy_changes.pdf",
             ("resource", "multiplier", "changed_action_fraction")),
    PlotSpec("Q9", "workload robustness", "q9_workload_robustness.pdf",
             ("workload", "shed_w", "binding_resources", "replay_fraction")),
    PlotSpec("Q9", "trace shift", "q9_trace_shift.pdf",
             ("workload", "input_case", "shed_w", "resume_p99_s")),
)


PROVENANCE_FIELDS = ("input_provenance", "result_provenance", "evidence_status")


def validate_rows(spec: PlotSpec, rows: list[dict]) -> None:
    required = set(spec.required_fields + PROVENANCE_FIELDS)
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError(f"{spec.question} {spec.name} lacks required fields")
    if any(row["evidence_status"] == "accepted"
           and "assumed" in row["input_provenance"].split("|") for row in rows):
        raise ValueError("assumed inputs cannot support accepted plot evidence")


def requirement_row(requirement, *, workload: str, sessions: int,
                    service_debt_replica_s: float = 0,
                    required_recovery_s: float = 0,
                    binding_resources: tuple[str, ...] = (),
                    input_provenance: str = "measured|assumed",
                    evidence_status: str = "sensitivity") -> dict:
    row = {
        "workload": workload,
        "sessions": sessions,
        "requested_shed_w": requirement.target_source_power_reduction_w,
        "shed_w": requirement.achieved_source_power_reduction_w,
        "achieved_shed_w": requirement.achieved_source_power_reduction_w,
        "unmet_shed_w": max(
            0, requirement.target_source_power_reduction_w
            - requirement.achieved_source_power_reduction_w,
        ),
        "route_bytes": requirement.wan_bytes,
        "transition_work": sum(requirement.destination_transition_work),
        "service_work": sum(requirement.destination_service_work),
        "kv_blocks": requirement.destination_kv_blocks,
        "service_debt_replica_s": service_debt_replica_s,
        "required_recovery_s": required_recovery_s,
        "binding_resources": "|".join(binding_resources),
        "input_provenance": input_provenance,
        "result_provenance": "simulated",
        "evidence_status": evidence_status,
    }
    validate_rows(next(spec for spec in PLOTS if spec.name == "requirement frontier"), [row])
    return row


def plot_requirement_frontier(rows: list[dict], out: Path) -> None:
    spec = next(spec for spec in PLOTS if spec.name == "requirement frontier")
    validate_rows(spec, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = (
        ("route_bytes", "Route bytes"),
        ("transition_work", "Transition replica-s"),
        ("service_work", "Ongoing replica-equivalents"),
        ("kv_blocks", "Live KV blocks"),
        ("service_debt_replica_s", "Queued replica-s"),
        ("required_recovery_s", "Required recovery (s)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for label in sorted({(row["workload"], row["sessions"]) for row in rows}):
        series = sorted(
            (row for row in rows
             if (row["workload"], row["sessions"]) == label),
            key=lambda row: row["shed_w"],
        )
        for axis, (field, ylabel) in zip(axes.flat, fields):
            axis.plot(
                [row["shed_w"] for row in series],
                [row[field] for row in series],
                marker="o", label=f"{label[0]} {label[1]:,}",
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Source accelerator power shed (W)")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def write(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation-manifest.json").write_text(json.dumps({
        "grid": evaluation_manifest(),
        "plots": [asdict(spec) for spec in PLOTS],
    }, indent=2, sort_keys=True) + "\n")
    with (out / "plot-specs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("question", "name", "filename", "required_fields"),
        )
        writer.writeheader()
        for spec in PLOTS:
            writer.writerow({
                **asdict(spec),
                "required_fields": "|".join(spec.required_fields),
            })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    write(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
