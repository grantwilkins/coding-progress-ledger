"""Prepare the combined training artifacts required by model work.

This writes the reproducible artifacts that Workstream I expects before
any fit path is allowed to run:
  - combined checkpoints / labels / manifest
  - D5 checkpoint construction audit
  - F1/F2/F3/F4 profile reports
  - F11 profiling go/no-go report
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from coding_estimator.checkpoints.build import write_combined_checkpoints
from coding_estimator.ingest.adapters import write_combined_manifest
from coding_estimator.labels.build import write_combined_labels
from coding_estimator.leakage.audit import build_audit, write_audit
from coding_estimator.models import assert_audit_clean
from coding_estimator.profile.checkpoints_distribution import write_distribution
from coding_estimator.profile.gate import run_gate, write_gate_report
from coding_estimator.profile.labels_balance import write_combined as write_labels_balance
from coding_estimator.profile.leakage import write_leakage_profile
from coding_estimator.profile.sources import write_source_profile


@dataclass(frozen=True)
class TrainingReadinessArtifacts:
    checkpoints_path: Path
    labels_path: Path
    manifest_path: Path
    audit_path: Path
    gate_report_path: Path
    source_profile_path: Path
    checkpoints_profile_path: Path
    labels_profile_path: Path
    leakage_profile_path: Path


def write_training_readiness_artifacts(
    *,
    dataset_dir: Path,
    reports_dir: Path,
    source_ids: Iterable[str] | None = None,
) -> TrainingReadinessArtifacts:
    """Write the combined datasets and the gating/profile artifacts."""
    manifests_dir = dataset_dir / "manifests"
    profiles_dir = dataset_dir / "profiles"
    selected_sources = list(source_ids) if source_ids is not None else None

    checkpoints_path, checkpoints_df = write_combined_checkpoints(
        dataset_dir / "checkpoints_all.parquet",
        source_ids=selected_sources,
    )
    labels_path, labels_df = write_combined_labels(
        dataset_dir,
        source_ids=selected_sources,
    )
    manifest_path, manifest_df = write_combined_manifest(
        manifests_dir,
        source_ids=selected_sources,
    )

    source_profile_path = write_source_profile(
        manifest_df,
        profiles_dir,
        checkpoints_df=checkpoints_df,
    )
    checkpoints_profile_path = write_distribution(checkpoints_df, profiles_dir)
    labels_profile_path = write_labels_balance(labels_df, profiles_dir)
    leakage_profile_path = write_leakage_profile(checkpoints_df, profiles_dir)

    audit = build_audit(
        checkpoints_df,
        sources=sorted(checkpoints_df["source"].unique()),
    )
    audit_path = write_audit(audit, reports_dir)

    gate_result = run_gate(labels_df=labels_df, checkpoints_df=checkpoints_df)
    gate_report_path = write_gate_report(gate_result, reports_dir)

    assert_audit_clean(audit_path)
    return TrainingReadinessArtifacts(
        checkpoints_path=checkpoints_path,
        labels_path=labels_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        gate_report_path=gate_report_path,
        source_profile_path=source_profile_path,
        checkpoints_profile_path=checkpoints_profile_path,
        labels_profile_path=labels_profile_path,
        leakage_profile_path=leakage_profile_path,
    )
