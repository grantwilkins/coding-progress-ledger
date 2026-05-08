"""Training-readiness artifact emission.

Claim:
    `write_training_readiness_artifacts()` writes the combined training
    inputs plus the D5/F11 gate artifacts required before Workstream I.

Plausible wrong implementations:
    - write only the markdown reports but not the combined parquet
      inputs future training code consumes
    - write the audit under a non-canonical filename, so
      `assert_audit_clean()` still fails
    - skip the manifest/profile artifacts, leaving the repo "trainable"
      but not auditable under its own rules
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coding_estimator.models import assert_audit_clean
from coding_estimator.models.readiness import write_training_readiness_artifacts


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_write_training_readiness_artifacts_emits_required_outputs(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"

    artifacts = write_training_readiness_artifacts(
        dataset_dir=dataset_dir,
        reports_dir=reports_dir,
    )

    assert artifacts.checkpoints_path == dataset_dir / "checkpoints_all.parquet"
    assert artifacts.labels_path == dataset_dir / "labels_all.parquet"
    assert artifacts.manifest_path == dataset_dir / "manifests" / "all_runs.csv"
    assert artifacts.audit_path == reports_dir / "CHECKPOINT_CONSTRUCTION_AUDIT.md"
    assert artifacts.gate_report_path == reports_dir / "F_profiling_go_no_go.md"

    for path in (
        artifacts.checkpoints_path,
        artifacts.labels_path,
        artifacts.manifest_path,
        artifacts.audit_path,
        artifacts.gate_report_path,
        artifacts.source_profile_path,
        artifacts.checkpoints_profile_path,
        artifacts.labels_profile_path,
        artifacts.leakage_profile_path,
    ):
        assert path.is_file(), path

    assert assert_audit_clean(artifacts.audit_path) == artifacts.audit_path

    gate_text = artifacts.gate_report_path.read_text(encoding="utf-8")
    assert "## Verdict: **PASS**" in gate_text

    checkpoints_df = pd.read_parquet(artifacts.checkpoints_path)
    assert set(checkpoints_df["source"].unique()) == {
        "hermes_pilot_h5_v2",
        "swe_agent_pilot",
        "tb_live",
    }


def test_write_training_readiness_artifacts_accepts_explicit_source_ids(
    real_ledger: None,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "datasets"
    reports_dir = tmp_path / "reports"

    artifacts = write_training_readiness_artifacts(
        dataset_dir=dataset_dir,
        reports_dir=reports_dir,
        source_ids=["tb_live_v2"],
    )

    checkpoints_df = pd.read_parquet(artifacts.checkpoints_path)
    labels_df = pd.read_parquet(artifacts.labels_path)
    manifest_df = pd.read_csv(artifacts.manifest_path)
    assert set(checkpoints_df["source"].unique()) == {"tb_live_v2"}
    assert set(labels_df["source"].unique()) == {"tb_live_v2"}
    assert set(manifest_df["source"].unique()) == {"tb_live_v2"}
    assert {"task_id", "task_family", "arm", "difficulty", "model_name"} <= set(
        checkpoints_df.columns
    )
    assert {"task_id", "task_family", "arm", "difficulty", "model_name"} <= set(
        labels_df.columns
    )
    assert {"task_id", "task_family", "arm", "difficulty", "model_name"} <= set(
        manifest_df.columns
    )
    assert manifest_df["model_name"].nunique() >= 3
