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
