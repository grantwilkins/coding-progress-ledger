from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolVersions:
    run_protocol_version: str = "0.1.0"
    artifact_layout_version: str = "0.1.0"
    observation_event_schema_version: str = "0.2.0"
    ledger_wire_schema_version: str = "1.0.0"
    checkpoint_schema_version: str = "0.1.0"
    label_schema_version: str = "0.1.0"
    estimator_feature_schema_version: str = "0.1.0"
    benchmark_adapter_version: str = "0.1.0"

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


VERSIONS = ProtocolVersions()


class RunStatus(StrEnum):
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_CRASH = "agent_crash"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_CRASH = "verifier_crash"
    DOCKER_BUILD_FAILURE = "docker_build_failure"
    ENVIRONMENT_SETUP_FAILURE = "environment_setup_failure"
    ARTIFACT_INCOMPLETE = "artifact_incomplete"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    QUARANTINED_LEAKAGE = "quarantined_leakage"


REQUIRED_RUN_ARTIFACTS = (
    "task.md",
    "task_metadata.json",
    "environment_manifest.json",
    "protocol_manifest.json",
    "transcript.jsonl",
    "observation_events.jsonl",
    "events.jsonl",
    "ledger.jsonl",
    "progress.csv",
    "progress_by_category.csv",
    "summary_by_category.json",
    "run_manifest.json",
    "verifier_output.txt",
    "run_notes.md",
)

ESTIMATOR_STAGE_ARTIFACTS = (
    "checkpoints.parquet",
    "labels.parquet",
    "estimator_predictions.parquet",
    "checkpoint_feature_manifest.json",
)

CONDITIONAL_ARTIFACTS = (
    "final_diff.patch",
    "patch_predictions.json",
    "container_logs.txt",
    "docker_build.log",
    "harbor_job_metadata.json",
    "swe_bench_patch.json",
)

TERMINAL_ANALYSIS_STATUSES = frozenset(
    {RunStatus.COMPLETED_SUCCESS, RunStatus.COMPLETED_FAILURE}
)
PROCESS_DYNAMICS_STATUSES = frozenset(
    {RunStatus.COMPLETED_SUCCESS, RunStatus.COMPLETED_FAILURE, RunStatus.AGENT_TIMEOUT}
)

PARTIAL_STATUS_REQUIRED_ARTIFACTS = (
    "task_metadata.json",
    "environment_manifest.json",
    "protocol_manifest.json",
    "run_manifest.json",
    "run_notes.md",
)

PROCESS_DYNAMICS_REQUIRED_ARTIFACTS = (
    "task.md",
    "task_metadata.json",
    "environment_manifest.json",
    "protocol_manifest.json",
    "transcript.jsonl",
    "observation_events.jsonl",
    "events.jsonl",
    "ledger.jsonl",
    "progress.csv",
    "progress_by_category.csv",
    "summary_by_category.json",
    "run_manifest.json",
    "run_notes.md",
)

VERIFIER_FAILURE_REQUIRED_ARTIFACTS = (
    *PROCESS_DYNAMICS_REQUIRED_ARTIFACTS,
    "verifier_output.txt",
)


def required_artifacts_for_status(status: RunStatus | str) -> tuple[str, ...]:
    status = RunStatus(status)
    if status in TERMINAL_ANALYSIS_STATUSES:
        return REQUIRED_RUN_ARTIFACTS
    if status in {RunStatus.AGENT_TIMEOUT, RunStatus.AGENT_CRASH}:
        return PROCESS_DYNAMICS_REQUIRED_ARTIFACTS
    if status in {RunStatus.VERIFIER_TIMEOUT, RunStatus.VERIFIER_CRASH}:
        return VERIFIER_FAILURE_REQUIRED_ARTIFACTS
    return PARTIAL_STATUS_REQUIRED_ARTIFACTS


def missing_artifacts(run_dir: Path, status: RunStatus | str) -> list[str]:
    return [
        name
        for name in required_artifacts_for_status(status)
        if not (run_dir / name).exists()
    ]


def analysis_inclusion(status: RunStatus | str) -> dict[str, bool]:
    status = RunStatus(status)
    return {
        "terminal_success": status in TERMINAL_ANALYSIS_STATUSES,
        "process_dynamics": status in PROCESS_DYNAMICS_STATUSES,
        "artifact_quality": True,
    }


def protocol_manifest_payload(
    *,
    coding_progress_ledger_sha: str,
    coding_estimator_sha: str,
    coding_data_collection_sha: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        **VERSIONS.to_dict(),
        "coding_progress_ledger_sha": coding_progress_ledger_sha,
        "coding_estimator_sha": coding_estimator_sha,
        "coding_data_collection_sha": coding_data_collection_sha,
    }
    if extra:
        payload.update(extra)
    return payload
