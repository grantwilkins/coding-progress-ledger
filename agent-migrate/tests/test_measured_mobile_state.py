from __future__ import annotations

import csv
from pathlib import Path

from agent_migrate_agent.measured_mobile_state import (
    layer_distribution,
    measured_restart_packages,
    measured_restart_pressure,
    read_snapshot_index,
    threshold_rows,
)


def _write_index(path: Path, *, clean_run_dir: str | None = None) -> None:
    rows = [
        {
            "run_id": "clean",
            "run_dir": clean_run_dir or str(path.parent / "clean"),
            "run_status": "completed_success",
            "final_success": "True",
            "eligible_for_l_gate": "True",
            "agent_workspace_retained": "True",
            "run_validation_passed": "",
            "leakage_passed": "True",
            "leakage_hit_count": "0",
            "row_usable_for_claims": "True",
            "clean_repo_bytes": "0",
            "clean_repo_bytes_provenance": "missing",
            "initial_workspace_bytes": "0",
            "initial_workspace_bytes_provenance": "missing",
            "final_workspace_bytes": "1000000000",
            "final_workspace_bytes_provenance": "measured",
            "unchanged_initial_bytes": "0",
            "unchanged_initial_bytes_provenance": "missing_without_initial_workspace_manifest",
            "modified_file_bytes": "1200",
            "modified_file_bytes_provenance": "patch_file_bytes",
            "new_file_bytes": "3000",
            "new_file_bytes_provenance": "trace_derived_touched_file_upper_bound",
            "deleted_file_bytes": "0",
            "deleted_file_bytes_provenance": "missing_without_initial_workspace_manifest",
            "final_diff_bytes": "1200",
            "final_diff_bytes_provenance": "measured",
            "touched_file_bytes": "3000",
            "touched_file_bytes_provenance": "trace_derived",
            "read_file_bytes": "4000",
            "read_file_bytes_provenance": "trace_derived",
            "tool_output_bytes": "500",
            "tool_output_bytes_provenance": "lower_bound_transcript_snippet_bytes",
            "test_log_bytes": "700",
            "test_log_bytes_provenance": "measured",
            "build_artifact_bytes": "0",
            "build_artifact_bytes_provenance": "measured",
            "dependency_cache_bytes": "0",
            "dependency_cache_bytes_provenance": "measured",
            "retrieved_document_bytes": "0",
            "retrieved_document_bytes_provenance": "measured",
            "workspace_total_bytes": "1000000000",
            "workspace_total_bytes_provenance": "measured",
            "hidden_or_protected_bytes": "0",
            "hidden_or_protected_bytes_provenance": "measured",
            "skipped_symlink_count": "0",
            "setup_command_count": "0",
            "lockfile_count": "1",
            "final_diff_semantics": "patch_file_bytes_not_touched_file_payload",
        },
        {
            "run_id": "quarantined",
            "run_dir": str(path.parent / "quarantined"),
            "run_status": "completed_success",
            "final_success": "True",
            "eligible_for_l_gate": "True",
            "agent_workspace_retained": "True",
            "leakage_passed": "False",
            "leakage_hit_count": "2",
            "row_usable_for_claims": "False",
            "final_workspace_bytes": "9000000000",
            "final_workspace_bytes_provenance": "measured",
            "modified_file_bytes": "9000000000",
            "modified_file_bytes_provenance": "patch_file_bytes",
            "final_diff_bytes": "9000000000",
            "final_diff_bytes_provenance": "measured",
            "tool_output_bytes": "0",
            "tool_output_bytes_provenance": "missing",
            "test_log_bytes": "0",
            "test_log_bytes_provenance": "missing",
            "workspace_total_bytes": "9000000000",
            "workspace_total_bytes_provenance": "measured",
            "hidden_or_protected_bytes": "1",
            "hidden_or_protected_bytes_provenance": "measured",
        },
    ]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_distribution_uses_claim_usable_rows_and_keeps_snapshot_as_representation_cost(tmp_path: Path) -> None:
    index = tmp_path / "raw_snapshot_index.csv"
    _write_index(index, clean_run_dir="runs/batch/clean")

    snapshots = read_snapshot_index(index)
    rows = {row.layer: row for row in layer_distribution(snapshots)}
    thresholds = {row.metric: row for row in threshold_rows(snapshots)}

    assert rows["full_workspace_snapshot"].p50_bytes == 1_000_000_000
    assert rows["modified_files_patch"].p50_bytes == 1_200
    assert rows["tool_outputs"].provenance_summary == "lower_bound_transcript_snippet_bytes"
    assert thresholds["dirty_payload_gt_1mb"].n_runs == 1
    assert thresholds["dirty_payload_gt_1mb"].n_over_threshold == 0


def test_measured_packages_compare_diff_against_full_snapshot_without_using_snapshot_as_headline(
    tmp_path: Path,
) -> None:
    index = tmp_path / "raw_snapshot_index.csv"
    _write_index(index)

    snapshots = read_snapshot_index(index)
    rows = {(row.run_id, row.package_type): row for row in measured_restart_packages(snapshots)}

    diff = rows[("clean", "base_repo_plus_diff")]
    full = rows[("clean", "full_workspace_snapshot")]
    assert diff.structurally_valid is True
    assert full.structurally_valid is True
    assert full.bytes_moved > diff.bytes_moved * 100_000
    assert ("quarantined", "full_workspace_snapshot") not in rows


def test_restart_pressure_reads_tokens_from_upstream_relative_snapshot_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "coding-data-collection"
    run_dir = source_root / "runs" / "batch" / "clean"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        '{"metrics": {"total_model_calls": 2, "total_tokens_in": 80000}}',
        encoding="utf-8",
    )
    index = tmp_path / "raw_snapshot_index.csv"
    _write_index(index, clean_run_dir="runs/batch/clean")

    snapshots = read_snapshot_index(index)
    rows, _ = measured_restart_pressure(
        snapshots,
        repo_root=Path(__file__).resolve().parents[1],
        source_root=source_root,
        max_workflows=1,
    )

    cache = next(row for row in rows if row.policy == "cache_reuse")
    assert cache.p50_resume_proxy_s > 0.2
