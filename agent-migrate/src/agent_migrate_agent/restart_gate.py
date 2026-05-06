"""Gate 3 minimal representation-aware restart table."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .cut_points import CutPoint, find_cut_points, load_trace_jsonl
from .resume_packages import (
    StateEntry,
    WorkspaceFileEntry,
    build_full_workspace_snapshot,
    build_prompt_only,
    build_transcript_plus_harness_state,
)
from .resume_validator import validate_package


@dataclass(frozen=True)
class MinimalRestartRow:
    trace_id: str
    workflow_id: str
    session_id: str
    event_index: int
    phase: str
    package_type: str
    structurally_valid: bool
    validation_reasons: str
    bytes_moved: int
    lazy_bytes_required_later: int
    model_resume_s: float
    environment_resume_s: float
    task_resume_s: float


def run_minimal_restart_gate(
    events: list[dict],
    cut_points: list[CutPoint],
    *,
    harness_config: dict,
    workspace_files: tuple[WorkspaceFileEntry, ...],
    base_commit: str = "synthetic_base_commit",
    diff_blob: str = "",
    max_cuts: int = 3,
) -> list[MinimalRestartRow]:
    rows: list[MinimalRestartRow] = []
    selected = cut_points[:max_cuts]
    for cut in selected:
        for package_type, package in (
            ("prompt_transcript_only", build_prompt_only(events, cut)),
            (
                "base_repo_plus_diff",
                _build_base_repo_plus_diff(
                    events, cut, harness_config=harness_config,
                    base_commit=base_commit, diff_blob=diff_blob,
                ),
            ),
            (
                "full_workspace_snapshot",
                build_full_workspace_snapshot(
                    events, cut, harness_config=harness_config,
                    workspace_files=workspace_files,
                ),
            ),
        ):
            result = validate_package(package, events)
            model_s = _model_resume_s(package.included_bytes)
            env_s = _environment_resume_s(package.included_bytes, package.lazy_rehydrate_bytes, result.valid)
            rows.append(MinimalRestartRow(
                trace_id=cut.trace_id,
                workflow_id=cut.workflow_id,
                session_id=cut.session_id,
                event_index=cut.event_index,
                phase=cut.phase,
                package_type=package_type,
                structurally_valid=result.valid,
                validation_reasons=";".join(result.reasons),
                bytes_moved=package.included_bytes,
                lazy_bytes_required_later=package.lazy_rehydrate_bytes,
                model_resume_s=model_s,
                environment_resume_s=env_s,
                task_resume_s=max(model_s, env_s),
            ))
    return rows


def write_minimal_restart_table(rows: list[MinimalRestartRow], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MinimalRestartRow.__dataclass_fields__.keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(repo_root: str | Path) -> None:
    repo = Path(repo_root)
    trace = repo / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl"
    events = load_trace_jsonl(trace)
    cuts = find_cut_points(events, trace_id="h5a")
    rows = run_minimal_restart_gate(
        events,
        cuts,
        harness_config={"cwd": "/workspace", "open_file": "", "env": {}},
        workspace_files=(),
        diff_blob=_synthetic_diff_blob(),
    )
    write_minimal_restart_table(
        rows,
        repo / "runs" / "restart_representation" / "minimal_package_table.csv",
    )


def _build_base_repo_plus_diff(
    events: list[dict],
    cut: CutPoint,
    *,
    harness_config: dict,
    base_commit: str,
    diff_blob: str,
):
    package = build_transcript_plus_harness_state(
        events,
        cut,
        harness_config=harness_config,
    )
    package = replace(package, base_commit=base_commit, diff_blob=diff_blob)
    entries = list(package.state_entries)
    for payload in _required_declared_non_prompt_states(events, cut.event_index):
        entries.append(StateEntry(
            state_id=payload.get("state_id", ""),
            layer=payload.get("layer", ""),
            bytes=0,
            content_hash=payload.get("content_hash", ""),
            materialization="included",
            validator="digest",
            role_at_cut=payload.get("role_at_cut"),
        ))
    entries.sort(key=lambda entry: entry.state_id)
    return replace(package, state_entries=tuple(entries))


def _required_declared_non_prompt_states(events: list[dict], cut_index: int) -> list[dict]:
    required = _next_call_reads(events, cut_index)
    out: list[dict] = []
    for event in events[:cut_index]:
        if event.get("event_type") != "state_declare":
            continue
        payload = event.get("payload") or {}
        if payload.get("state_id") in required and payload.get("layer") != "prompt_context":
            out.append(payload)
    return out


def _next_call_reads(events: list[dict], cut_index: int) -> set[str]:
    if cut_index >= len(events):
        return set()
    head = events[cut_index]
    consumer = head.get("subtask_id")
    session_id = (head.get("payload") or {}).get("session_id")
    matched: set[str] = set()
    all_reads: set[str] = set()
    for event in events[cut_index + 1:]:
        if (
            event.get("event_type") == "add_subtask"
            and (event.get("payload") or {}).get("node_type") == "llm_call"
            and (event.get("payload") or {}).get("session_id") == session_id
        ):
            break
        if event.get("event_type") != "state_read":
            continue
        payload = event.get("payload") or {}
        state_id = payload.get("state_id")
        if not state_id:
            continue
        all_reads.add(state_id)
        if payload.get("consumer_node_id") == consumer:
            matched.add(state_id)
    return matched or all_reads


def _model_resume_s(bytes_moved: int) -> float:
    return bytes_moved / 5e9


def _environment_resume_s(bytes_moved: int, lazy_bytes: int, valid: bool) -> float:
    if not valid:
        return float("inf")
    return (bytes_moved + lazy_bytes) / 1e9


def _synthetic_diff_blob() -> str:
    return (
        "diff --git a/workspace.txt b/workspace.txt\n"
        "index e69de29..1269488 100644\n"
        "--- a/workspace.txt\n"
        "+++ b/workspace.txt\n"
        "@@ -0,0 +1 @@\n"
        "+synthetic workspace diff representation\n"
    )


__all__ = [
    "MinimalRestartRow",
    "main",
    "run_minimal_restart_gate",
    "write_minimal_restart_table",
]
