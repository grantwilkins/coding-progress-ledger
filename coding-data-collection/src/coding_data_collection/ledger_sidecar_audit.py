from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .observation import read_jsonl
from .validation import validate_run_dir


SIDECAR_ARTIFACTS = (
    "ledger.jsonl",
    "progress.csv",
    "progress_by_category.csv",
    "summary_by_category.json",
)

COLLECTION_CONTROL_ARTIFACTS = (
    "task_metadata.json",
    "environment_manifest.json",
    "protocol_manifest.json",
    "run_manifest.json",
)


def ledger_sidecar_report(run_dir: Path) -> dict[str, Any]:
    events = read_jsonl(run_dir / "events.jsonl")
    issues: list[str] = []
    collection_validation_issues = [
        f"{issue.artifact}: {issue.message}"
        for issue in validate_run_dir(run_dir)
    ]
    issues.extend(collection_validation_issues)
    missing_collection_controls = [
        artifact
        for artifact in COLLECTION_CONTROL_ARTIFACTS
        if not (run_dir / artifact).is_file()
    ]
    issues.extend(
        f"{artifact}: missing collection control artifact"
        for artifact in missing_collection_controls
    )
    generated = {}
    for artifact in SIDECAR_ARTIFACTS:
        path = run_dir / artifact
        generated[artifact] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
        if not path.is_file():
            issues.append(f"{artifact}: missing")
        elif path.stat().st_size == 0:
            issues.append(f"{artifact}: empty")

    summary = _read_json(run_dir / "summary_by_category.json")
    ledger_path = run_dir / "ledger.jsonl"
    ledger_sha = _sha256_file(ledger_path) if ledger_path.is_file() else None
    if summary.get("generator") != "ledger_progress.sidecar":
        issues.append("summary_by_category.json: generator is not ledger_progress.sidecar")
    if ledger_sha and summary.get("source_ledger_sha256") != ledger_sha:
        issues.append("summary_by_category.json: source_ledger_sha256 does not match ledger.jsonl")
    if not events:
        issues.append("events.jsonl: no wire events")

    ledger_events = read_jsonl(ledger_path)
    if ledger_events and ledger_events[0].get("event_type") != "init":
        issues.append("ledger.jsonl: first event is not init")

    progress_rows = _read_csv(run_dir / "progress.csv")
    progress_by_category_rows = _read_csv(run_dir / "progress_by_category.csv")
    if len(progress_rows) < 2:
        issues.append("progress.csv: no progress rows")
    if len(progress_by_category_rows) < 2:
        issues.append("progress_by_category.csv: no progress rows")
    elif "coding_progress" not in progress_by_category_rows[0]:
        issues.append("progress_by_category.csv: missing coding_progress column")

    return {
        "run_dir": str(run_dir),
        "event_count": len(events),
        "ledger_event_count": len(ledger_events),
        "generated_artifacts": generated,
        "summary_generator": summary.get("generator"),
        "source_ledger_sha256_matches": bool(ledger_sha and summary.get("source_ledger_sha256") == ledger_sha),
        "progress_rows": max(len(progress_rows) - 1, 0),
        "progress_by_category_rows": max(len(progress_by_category_rows) - 1, 0),
        "issues": issues,
        "collection_run_valid": not collection_validation_issues and not missing_collection_controls,
        "collection_validation_issues": collection_validation_issues,
        "missing_collection_control_artifacts": missing_collection_controls,
        "passed": not issues,
    }


def corpus_ledger_sidecar_report(run_dirs: list[Path]) -> dict[str, Any]:
    reports = [ledger_sidecar_report(run_dir) for run_dir in run_dirs]
    return {
        "run_count": len(reports),
        "passed": all(report["passed"] for report in reports),
        "runs": reports,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
