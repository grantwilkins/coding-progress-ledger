from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .observation import read_jsonl
from .protocol import RunStatus, VERSIONS, analysis_inclusion, missing_artifacts


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"

JSON_SCHEMAS = {
    "run_manifest.json": "run_manifest.schema.json",
    "task_metadata.json": "task_metadata.schema.json",
    "environment_manifest.json": "environment_manifest.schema.json",
    "protocol_manifest.json": "protocol_manifest.schema.json",
}

JSONL_SCHEMAS = {
    "observation_events.jsonl": "observation_event.schema.json",
    "events.jsonl": "ledger_wire_event.schema.json",
}


@dataclass(frozen=True)
class ValidationIssue:
    artifact: str
    message: str


class RunValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        joined = "\n".join(f"{issue.artifact}: {issue.message}" for issue in issues)
        super().__init__(joined)


def validate_run_dir(run_dir: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    run_manifest = _read_json(run_dir / "run_manifest.json")
    status_text = run_manifest.get("run_status")

    if status_text:
        try:
            status = RunStatus(status_text)
            for name in missing_artifacts(run_dir, status):
                issues.append(ValidationIssue(name, f"missing for status {status.value}"))
        except ValueError:
            issues.append(ValidationIssue("run_manifest.json", f"unknown run_status {status_text!r}"))
    else:
        status = None

    for artifact, schema_name in JSON_SCHEMAS.items():
        path = run_dir / artifact
        if path.exists():
            issues.extend(_validate_json(path, _load_schema(schema_name)))

    for artifact, schema_name in JSONL_SCHEMAS.items():
        path = run_dir / artifact
        if path.exists():
            issues.extend(_validate_jsonl(path, _load_schema(schema_name)))

    if run_manifest:
        issues.extend(_validate_manifest_semantics(run_manifest, status))
    issues.extend(_validate_observation_timing(run_dir))
    return issues


def assert_valid_run_dir(run_dir: Path) -> None:
    issues = validate_run_dir(run_dir)
    if issues:
        raise RunValidationError(issues)


def _validate_manifest_semantics(
    run_manifest: dict[str, Any],
    status: RunStatus | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if status is None:
        return issues
    expected = analysis_inclusion(status)
    observed = run_manifest.get("analysis_inclusion")
    if observed != expected:
        issues.append(
            ValidationIssue(
                "run_manifest.json",
                f"analysis_inclusion must be {expected} for {status.value}",
            )
        )
    if status in {RunStatus.COMPLETED_SUCCESS, RunStatus.COMPLETED_FAILURE}:
        expected_success = status == RunStatus.COMPLETED_SUCCESS
        if run_manifest.get("final_success") is not expected_success:
            issues.append(
                ValidationIssue(
                    "run_manifest.json",
                    f"final_success must be {expected_success} for {status.value}",
                )
            )
    elif run_manifest.get("analysis_inclusion", {}).get("terminal_success") is True:
        issues.append(
            ValidationIssue(
                "run_manifest.json",
                f"{status.value} cannot be included in terminal-success analysis",
            )
        )
    return issues


def _validate_observation_timing(run_dir: Path) -> list[ValidationIssue]:
    transcript = read_jsonl(run_dir / "transcript.jsonl")
    observations = read_jsonl(run_dir / "observation_events.jsonl")
    if not transcript or not observations:
        return []
    max_transcript_step = max(int(row.get("step", 0)) for row in transcript)
    terminal_types = {"verifier_pass", "verifier_fail", "verifier_disagreement", "expected_file_missing"}
    issues: list[ValidationIssue] = []
    for line_number, event in enumerate(observations, 1):
        visible = event.get("payload", {}).get("visible_to_agent")
        if event.get("event_type") in terminal_types and int(event.get("step", -1)) <= max_transcript_step:
            issues.append(
                ValidationIssue(
                    "observation_events.jsonl",
                    f"line {line_number}: post-terminal event at transcript step {event.get('step')}",
                )
            )
        if event.get("event_type") in terminal_types and visible is not False:
            issues.append(
                ValidationIssue(
                    "observation_events.jsonl",
                    f"line {line_number}: terminal event must not be visible_to_agent",
                )
            )
        if event.get("source_artifact") in {"verifier_output.txt", "oracle_workspace_snapshot"} and visible is not False:
            issues.append(
                ValidationIssue(
                    "observation_events.jsonl",
                    f"line {line_number}: hidden phase event must not be visible_to_agent",
                )
            )
    return issues


def _validate_json(path: Path, schema: dict[str, Any]) -> list[ValidationIssue]:
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        return [ValidationIssue(path.name, f"invalid JSON: {exc}")]
    return _schema_issues(path.name, payload, schema)


def _validate_jsonl(path: Path, schema: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(path.name, f"line {line_number}: invalid JSON: {exc}"))
            continue
        for issue in _schema_issues(path.name, payload, schema):
            issues.append(ValidationIssue(issue.artifact, f"line {line_number}: {issue.message}"))
    return issues


def _schema_issues(artifact: str, payload: Any, schema: dict[str, Any]) -> list[ValidationIssue]:
    validator = Draft202012Validator(schema)
    issues: list[ValidationIssue] = []
    for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path)
        prefix = f"{location}: " if location else ""
        issues.append(ValidationIssue(artifact, prefix + error.message))
    return issues


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def expected_protocol_versions() -> dict[str, str]:
    return VERSIONS.to_dict()
