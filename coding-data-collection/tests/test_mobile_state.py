"""
Claim:
Mobile-state snapshots measure retained run bytes without link inflation, find
nested run artifacts, and label patch/snippet semantics honestly.

Plausible wrong implementations:
- Only scan one directory level below runs/ and miss real batch/run layouts.
- Count hardlinked files twice or follow symlinks outside the retained workspace.
- Treat final_diff.patch bytes as touched-file payload bytes.
- Drop transcript tool-output snippets from the mobile prompt-state budget.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from coding_data_collection.mobile_state import snapshot_run, snapshot_run_roots, write_snapshots


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_snapshot_run_counts_links_and_transcript_bytes_correctly(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "batch" / "task__model"
    workspace = run / "agent_workspace"
    workspace.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({
            "run_id": "task__model",
            "run_status": "completed_failure",
            "final_success": False,
            "metrics": {"eligible_for_L_gate": True},
        }),
        encoding="utf-8",
    )
    (workspace / "answer.txt").write_text("abcd", encoding="utf-8")
    os.link(workspace / "answer.txt", workspace / "answer-hardlink.txt")
    (workspace / "outside-link").symlink_to(tmp_path / "outside.txt")
    (run / "final_diff.patch").write_text("patch-bytes", encoding="utf-8")
    _jsonl(
        run / "transcript.jsonl",
        [
            {"kind": "write_file", "path": "answer.txt"},
            {"kind": "write_file", "path": "../outside.txt"},
            {"kind": "read_file", "path": "/tmp/outside.txt", "stdout_snippet": "ignored"},
            {"kind": "read_file", "path": "answer.txt", "stdout_snippet": "abcd"},
            {"kind": "shell", "command": "pytest -q", "stdout_snippet": "ok", "stderr_snippet": ""},
        ],
    )

    snapshot = snapshot_run(run)

    assert snapshot.workspace_total_bytes == 4
    assert snapshot.final_workspace_bytes == 4
    assert snapshot.initial_workspace_bytes == 0
    assert snapshot.initial_workspace_bytes_provenance == "missing"
    assert snapshot.unchanged_initial_bytes_provenance == "missing_without_initial_workspace_manifest"
    assert snapshot.touched_file_bytes == 4
    assert snapshot.new_file_bytes == 4
    assert snapshot.new_file_bytes_provenance == "trace_derived_touched_file_upper_bound"
    assert snapshot.read_file_bytes == 4
    assert snapshot.skipped_symlink_count == 1
    assert snapshot.tool_output_bytes == len("ignoredabcdok".encode("utf-8"))
    assert snapshot.tool_output_bytes_provenance == "lower_bound_transcript_snippet_bytes"
    assert snapshot.final_diff_bytes == len("patch-bytes")
    assert snapshot.modified_file_bytes == len("patch-bytes")
    assert snapshot.modified_file_bytes_provenance == "patch_file_bytes"
    assert snapshot.final_diff_semantics == "patch_file_bytes_not_touched_file_payload"
    assert snapshot.row_usable_for_claims is True


def test_snapshot_run_classifies_build_cache_as_recomputable_not_hidden(tmp_path: Path) -> None:
    run = tmp_path / "run"
    workspace = run / "agent_workspace" / "__pycache__"
    workspace.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (workspace / "x.pyc").write_bytes(b"12345")

    snapshot = snapshot_run(run)

    assert snapshot.build_artifact_bytes == 5
    assert snapshot.hidden_or_protected_bytes == 0
    assert snapshot.leakage_passed is True


def test_snapshot_run_roots_recurses_into_batch_run_layout(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "batch" / "task__model"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "nested"}), encoding="utf-8")

    snapshots = snapshot_run_roots([tmp_path / "runs"])

    assert [snapshot.run_id for snapshot in snapshots] == ["nested"]


def test_hidden_or_protected_workspace_quarantines_claim_row(tmp_path: Path) -> None:
    run = tmp_path / "run"
    workspace = run / "agent_workspace" / "tests"
    workspace.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (workspace / "test_secret.py").write_text("benchmark canary", encoding="utf-8")

    snapshot = snapshot_run(run)

    assert snapshot.hidden_or_protected_bytes > 0
    assert snapshot.leakage_passed is False
    assert snapshot.leakage_hit_count >= 1
    assert snapshot.row_usable_for_claims is False


def test_missing_workspace_is_explicitly_missing_and_not_claim_usable(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "missing-ws"}), encoding="utf-8")

    snapshot = snapshot_run(run)

    assert snapshot.agent_workspace_retained is False
    assert snapshot.final_workspace_bytes == 0
    assert snapshot.final_workspace_bytes_provenance == "missing"
    assert snapshot.leakage_passed is None
    assert snapshot.row_usable_for_claims is False


def test_snapshot_json_does_not_emit_file_contents(tmp_path: Path) -> None:
    run = tmp_path / "run"
    workspace = run / "agent_workspace"
    workspace.mkdir(parents=True)
    secret = "secret-content-that-must-not-be-exported"
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "safe"}), encoding="utf-8")
    (workspace / "answer.txt").write_text(secret, encoding="utf-8")

    out = tmp_path / "out"
    write_snapshots([snapshot_run(run)], out)

    combined = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir())
    assert secret not in combined
    assert "answer.txt" not in combined
