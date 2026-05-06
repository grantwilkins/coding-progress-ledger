"""C4 — static cut-and-resume ablation table.

For each cut point and resume package shape, build the package, run the C3
static validator, and emit CSV-ready rows. This module does not execute a
harness, run tests, call a model, or run tools. Diff packages may invoke C3's
syntactic git checks only when a caller supplies `base_repo_path`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .cut_points import CutPoint
from .resume_packages import (
    ResumePackage,
    WorkspaceFileEntry,
    build_agent_migrate_minimal,
    build_full_workspace_snapshot,
    build_prompt_only,
    build_transcript_plus_diff,
    build_transcript_plus_harness_state,
)
from .resume_validator import validate_package


@dataclass(frozen=True)
class AblationRow:
    trace_id: str
    workflow_id: str
    session_id: str
    event_index: int
    phase: str
    package_type: str
    valid: bool
    validation_reasons: str
    checks_run: str
    bytes_moved: int
    bytes_lazy_rehydrate_target: int
    extra_setup_steps_recorded: int
    dominant_resource: str
    k4_ran: bool


def build_packages_for_cut(
    events: list[dict],
    cut_point: CutPoint,
    *,
    harness_config: dict,
    workspace_files: Iterable[WorkspaceFileEntry] = (),
    workspace_layer_for_file: dict[str, str] | None = None,
    base_commit: str = "",
    diff_blob: str = "",
) -> tuple[ResumePackage, ...]:
    files = tuple(workspace_files)
    return (
        build_prompt_only(events, cut_point),
        build_transcript_plus_harness_state(events, cut_point, harness_config=harness_config),
        build_transcript_plus_diff(
            events,
            cut_point,
            harness_config=harness_config,
            base_commit=base_commit,
            diff_blob=diff_blob,
        ),
        build_full_workspace_snapshot(events, cut_point, harness_config=harness_config, workspace_files=files),
        build_agent_migrate_minimal(
            events,
            cut_point,
            harness_config=harness_config,
            workspace_files=files,
            workspace_layer_for_file=workspace_layer_for_file,
        ),
    )


def run_resume_ablation(
    events: list[dict],
    cut_points: Iterable[CutPoint],
    *,
    harness_config: dict,
    workspace_files: Iterable[WorkspaceFileEntry] = (),
    workspace_layer_for_file: dict[str, str] | None = None,
    base_commit: str = "",
    diff_blob: str = "",
    base_repo_path: str | Path | None = None,
) -> list[AblationRow]:
    files = tuple(workspace_files)
    rows: list[AblationRow] = []
    for cut in cut_points:
        packages = build_packages_for_cut(
            events,
            cut,
            harness_config=harness_config,
            workspace_files=files,
            workspace_layer_for_file=workspace_layer_for_file,
            base_commit=base_commit,
            diff_blob=diff_blob,
        )
        for package in packages:
            result = validate_package(package, events, base_repo_path=base_repo_path)
            rows.append(AblationRow(
                trace_id=cut.trace_id,
                workflow_id=cut.workflow_id,
                session_id=cut.session_id,
                event_index=cut.event_index,
                phase=cut.phase,
                package_type=package.package_type,
                valid=result.valid,
                validation_reasons=";".join(result.reasons),
                checks_run=";".join(result.checks_run),
                bytes_moved=package.included_bytes,
                bytes_lazy_rehydrate_target=package.lazy_rehydrate_bytes,
                extra_setup_steps_recorded=count_extra_setup_steps(package),
                dominant_resource="",
                k4_ran=False,
            ))
    return rows


def count_extra_setup_steps(package: ResumePackage) -> int:
    steps = 0
    if package.harness_config is not None:
        steps += 1
    if package.diff_blob:
        steps += 1
    steps += sum(
        1
        for entry in package.state_entries
        if entry.materialization in {"lazy_rehydrate", "globally_available"}
    )
    return steps


def write_ablation_csv(rows: Iterable[AblationRow], out_path: str | Path) -> None:
    rows = list(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AblationRow.__dataclass_fields__.keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
