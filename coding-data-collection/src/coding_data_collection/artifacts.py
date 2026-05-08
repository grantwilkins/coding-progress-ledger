from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import RunStatus, VERSIONS, analysis_inclusion, missing_artifacts, protocol_manifest_payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_protocol_manifest(
    run_dir: Path,
    *,
    collection_root: Path,
    ledger_root: Path,
    estimator_root: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = protocol_manifest_payload(
        coding_progress_ledger_sha=git_sha(ledger_root),
        coding_estimator_sha=git_sha(estimator_root),
        coding_data_collection_sha=git_sha(collection_root),
        extra=extra,
    )
    path = run_dir / "protocol_manifest.json"
    write_json(path, payload)
    return path


def write_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    run_status: RunStatus | str,
    final_success: bool | None,
    termination_reason: str,
    metrics: dict[str, Any] | None = None,
) -> Path:
    status = RunStatus(run_status)
    payload = {
        "run_protocol_version": VERSIONS.run_protocol_version,
        "artifact_layout_version": VERSIONS.artifact_layout_version,
        "run_id": run_id,
        "run_status": status.value,
        "final_success": final_success,
        "termination_reason": termination_reason,
        "analysis_inclusion": analysis_inclusion(status),
        "created_at": utc_now(),
        "metrics": metrics or {},
    }
    path = run_dir / "run_manifest.json"
    write_json(path, payload)
    return path


def artifact_completeness_report(run_dir: Path, status: RunStatus | str) -> dict[str, Any]:
    missing = missing_artifacts(run_dir, status)
    return {
        "run_dir": str(run_dir),
        "run_status": RunStatus(status).value,
        "missing_artifacts": missing,
        "complete": not missing,
    }
