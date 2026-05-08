from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ESTIMATOR_ARTIFACTS = (
    "checkpoints.parquet",
    "labels.parquet",
    "estimator_predictions.parquet",
    "checkpoint_feature_manifest.json",
)


@dataclass(frozen=True)
class EstimatorBuildResult:
    corpus_id: str
    source_id: str
    artifact_dir: Path
    staged_runs_dir: Path
    manifest_path: Path
    validation_report: dict[str, Any]


def stage_runs_for_estimator(
    *,
    run_dirs: list[Path],
    estimator_root: Path,
    source_id: str,
    replace: bool = False,
) -> Path:
    staged_root = estimator_root / "runs" / source_id
    if replace and staged_root.exists():
        for child in staged_root.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    staged_root.mkdir(parents=True, exist_ok=True)
    for run_dir in run_dirs:
        source = run_dir.resolve()
        target = staged_root / run_dir.name
        if target.exists() or target.is_symlink():
            if not replace:
                continue
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        target.symlink_to(source, target_is_directory=True)
    return staged_root


def write_estimator_source_manifest(
    *,
    corpus_id: str,
    source_id: str,
    run_dirs: list[Path],
    artifact_dir: Path,
    staged_runs_dir: Path,
) -> Path:
    payload = {
        "corpus_id": corpus_id,
        "source_id": source_id,
        "staged_runs_dir": str(staged_runs_dir),
        "artifact_dir": str(artifact_dir),
        "run_count": len(run_dirs),
        "run_ids": [run_dir.name for run_dir in run_dirs],
        "artifact_contract": list(ESTIMATOR_ARTIFACTS),
        "producer": "coding-estimator",
    }
    path = artifact_dir / "estimator_source_manifest.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_estimator_artifact_builder(
    *,
    estimator_root: Path,
    source_id: str,
    artifact_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build_collection_artifacts.py",
            "--source",
            source_id,
            "--out-dir",
            str(artifact_dir.resolve()),
        ],
        cwd=estimator_root,
        text=True,
        capture_output=True,
    )


def validate_estimator_artifacts(artifact_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    artifact_status = {}
    for artifact in ESTIMATOR_ARTIFACTS:
        path = artifact_dir / artifact
        artifact_status[artifact] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
        if not path.is_file():
            issues.append(f"{artifact}: missing")
        elif path.stat().st_size == 0:
            issues.append(f"{artifact}: empty")

    checkpoint_rows = 0
    provenance_complete = False
    if (artifact_dir / "checkpoints.parquet").is_file():
        checkpoints = pd.read_parquet(artifact_dir / "checkpoints.parquet")
        checkpoint_rows = int(len(checkpoints))
        required = {
            "checkpoint_step",
            "max_ledger_step_used",
            "max_observation_step_used",
        }
        missing_columns = sorted(required - set(checkpoints.columns))
        if missing_columns:
            issues.append(f"checkpoints.parquet: missing provenance columns {missing_columns}")
        elif checkpoints.empty:
            issues.append("checkpoints.parquet: no checkpoint rows")
        else:
            null_counts = checkpoints[list(required)].isna().sum()
            for column, count in null_counts.items():
                if int(count) != 0:
                    issues.append(f"checkpoints.parquet: {column} has {int(count)} null values")
            future_ledger = checkpoints["max_ledger_step_used"] > checkpoints["checkpoint_step"]
            future_obs = checkpoints["max_observation_step_used"] > checkpoints["checkpoint_step"]
            if bool(future_ledger.any()):
                issues.append("checkpoints.parquet: max_ledger_step_used exceeds checkpoint_step")
            if bool(future_obs.any()):
                issues.append("checkpoints.parquet: max_observation_step_used exceeds checkpoint_step")
            provenance_complete = not any(
                issue.startswith("checkpoints.parquet:")
                for issue in issues
            )

    return {
        "artifact_dir": str(artifact_dir),
        "artifacts": artifact_status,
        "checkpoint_rows": checkpoint_rows,
        "prefix_provenance_complete": provenance_complete,
        "issues": issues,
        "passed": not issues,
    }


def build_estimator_artifacts(
    *,
    corpus_id: str,
    source_id: str,
    run_dirs: list[Path],
    estimator_root: Path,
    artifact_dir: Path,
    replace_staged_runs: bool = False,
) -> EstimatorBuildResult:
    staged_runs_dir = stage_runs_for_estimator(
        run_dirs=run_dirs,
        estimator_root=estimator_root,
        source_id=source_id,
        replace=replace_staged_runs,
    )
    manifest_path = write_estimator_source_manifest(
        corpus_id=corpus_id,
        source_id=source_id,
        run_dirs=run_dirs,
        artifact_dir=artifact_dir,
        staged_runs_dir=staged_runs_dir,
    )
    proc = run_estimator_artifact_builder(
        estimator_root=estimator_root,
        source_id=source_id,
        artifact_dir=artifact_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    validation_report = validate_estimator_artifacts(artifact_dir)
    return EstimatorBuildResult(
        corpus_id=corpus_id,
        source_id=source_id,
        artifact_dir=artifact_dir,
        staged_runs_dir=staged_runs_dir,
        manifest_path=manifest_path,
        validation_report=validation_report,
    )
