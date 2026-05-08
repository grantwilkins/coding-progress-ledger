"""
Claim:
Pre-pilot hardening must reject agent-visible leakage while allowing hidden
post-agent verifier/oracle artifacts, sample validation-attempt precision at
the event level, and block scale when pilot gates are not met.

Plausible wrong implementations:
- Scan every transcript row as agent-visible and fail valid hidden verifier
  rows, or skip transcript scanning entirely.
- Treat any validation_attempt event as precise without checking source text.
- Report corpus artifact completeness without failing missing run manifests.
- Let a short, low-observation smoke corpus pass the pilot gate report.
- Produce a failure analysis that does not block Workstream M.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coding_data_collection.audits import (
    corpus_artifact_audit_report,
    redaction_audit_report,
    validation_attempt_precision_report,
)
from coding_data_collection.estimator_artifacts import ESTIMATOR_ARTIFACTS
from coding_data_collection.observation import write_jsonl
from coding_data_collection.pilot_gates import failure_analysis, pilot_gate_report


def test_redaction_audit_scans_only_agent_visible_prompt_and_transcript(tmp_path: Path) -> None:
    (tmp_path / "task.md").write_text("Visible task text.\n", encoding="utf-8")
    write_jsonl(
        tmp_path / "transcript.jsonl",
        [
            {
                "step": 1,
                "kind": "shell",
                "summary": "agent_phase",
                "command": "docker run image bash -lc '[ ! -e /task/tests/test_outputs.py ]'",
                "stdout_snippet": "ordinary output",
            },
            {
                "step": 2,
                "kind": "shell",
                "summary": "verifier_phase",
                "stdout_snippet": "hidden verifier mentions test_outputs.py",
            },
        ],
    )
    write_jsonl(tmp_path / "events.jsonl", [])
    write_jsonl(tmp_path / "observation_events.jsonl", [])

    assert redaction_audit_report(tmp_path)["passed"] is True

    write_jsonl(
        tmp_path / "transcript.jsonl",
        [
            {
                "step": 1,
                "kind": "shell",
                "summary": "agent_phase",
                "stdout_snippet": "I read the oracle solution.",
            }
        ],
    )

    report = redaction_audit_report(tmp_path)
    assert report["passed"] is False
    assert report["redaction_hits"][0]["artifact"] == "transcript.jsonl"


def test_redaction_audit_scans_visible_event_and_observation_payloads(tmp_path: Path) -> None:
    (tmp_path / "task.md").write_text("Visible task text.\n", encoding="utf-8")
    write_jsonl(tmp_path / "transcript.jsonl", [])
    write_jsonl(
        tmp_path / "events.jsonl",
        [
            {
                "step": 1,
                "agent_step": {
                    "step": 1,
                    "kind": "shell",
                    "command": "cat solution.sh",
                    "stdout_snippet": "",
                },
            }
        ],
    )
    write_jsonl(
        tmp_path / "observation_events.jsonl",
        [
            {
                "step": 1,
                "event_type": "oracle_artifact_read",
                "payload": {"visible_to_agent": True, "path": "tests/test_outputs.py"},
            },
            {
                "step": 2,
                "event_type": "validation_attempt",
                "payload": {"visible_to_agent": False, "path": "tests/test_outputs.py"},
            },
        ],
    )

    report = redaction_audit_report(tmp_path)

    assert report["passed"] is False
    assert {hit["artifact"] for hit in report["redaction_hits"]} == {
        "events.jsonl",
        "observation_events.jsonl",
    }


def test_validation_attempt_precision_sample_rejects_unexplained_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(
        run_dir / "transcript.jsonl",
        [
            {"step": 1, "kind": "shell", "command": "pytest -q", "stdout_snippet": "1 passed"},
            {"step": 2, "kind": "shell", "command": "echo hello", "stdout_snippet": "hello"},
            {"step": 3, "kind": "shell", "command": "python -m pytest", "stdout_snippet": "2 passed"},
        ],
    )
    write_jsonl(
        run_dir / "observation_events.jsonl",
        [
            {"step": 1, "event_type": "validation_attempt", "payload": {"visible_to_agent": True}},
            {"step": 2, "event_type": "validation_attempt", "payload": {"visible_to_agent": True}},
        ],
    )

    report = validation_attempt_precision_report([run_dir])

    assert report["validation_attempt_count"] == 2
    assert report["sample_precision"] == 0.5
    assert report["likely_missed_validation_attempt_count"] == 1
    assert report["passed"] is False


def test_corpus_artifact_audit_fails_missing_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = corpus_artifact_audit_report([run_dir])

    assert report["passed"] is False
    assert report["runs"][0]["missing_artifacts"] == ["run_manifest.json"]


def test_pilot_gate_report_blocks_short_low_signal_corpus(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    estimator_dir = tmp_path / "estimator"
    run_dir.mkdir()
    _write_minimal_low_signal_run(run_dir)
    _write_minimal_estimator_artifacts(estimator_dir)

    report = pilot_gate_report([run_dir], estimator_artifact_dir=estimator_dir)
    analysis = failure_analysis(report)

    assert report["passed"] is False
    assert report["gate_inputs"]["eligible_run_count"] == 1
    assert report["gates"]["median_transcript_steps"] is False
    assert report["gates"]["median_observation_events_per_run"] is False
    assert report["gates"]["verifier_outcomes_reproducible"] is True
    assert "Do not run Workstream M" in analysis


def test_pilot_gate_report_excludes_protocol_smoke_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    estimator_dir = tmp_path / "estimator"
    run_dir.mkdir()
    _write_minimal_low_signal_run(run_dir, eligible_for_l_gate=False)
    _write_minimal_estimator_artifacts(estimator_dir)

    report = pilot_gate_report([run_dir], estimator_artifact_dir=estimator_dir)
    analysis = failure_analysis(report)

    assert report["gate_inputs"]["eligible_run_count"] == 0
    assert report["gate_inputs"]["excluded_protocol_smoke_run_count"] == 1
    assert report["gates"]["real_agent_pilot_runs_present"] is False
    assert report["run_metrics"] == []
    assert "protocol-smoke shell runs are excluded" in analysis


def _write_minimal_low_signal_run(run_dir: Path, *, eligible_for_l_gate: bool = True) -> None:
    write_jsonl(
        run_dir / "transcript.jsonl",
        [
            {"step": 1, "kind": "shell", "summary": "agent_phase", "command": "pytest -q", "exit_code": 0},
            {"step": 2, "kind": "done"},
        ],
    )
    write_jsonl(
        run_dir / "observation_events.jsonl",
        [
            {"step": 1, "event_type": "validation_attempt", "payload": {"visible_to_agent": True}},
            {"step": 3, "event_type": "verifier_pass", "payload": {"visible_to_agent": False}},
        ],
    )
    (run_dir / "progress_by_category.csv").write_text(
        "step,coding_progress\n0,0.0\n1,0.0\n2,1.0\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_status": "completed_success",
                "final_success": True,
                "metrics": {
                    "agent_backend": "model_tool_loop" if eligible_for_l_gate else "shell_command",
                    "pilot_type": "real_agent_pilot" if eligible_for_l_gate else "protocol_smoke",
                    "eligible_for_L_gate": eligible_for_l_gate,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "task.md").write_text("Task\n", encoding="utf-8")
    (run_dir / "verifier_determinism_report.json").write_text(
        json.dumps({"deterministic": True}),
        encoding="utf-8",
    )


def _write_minimal_estimator_artifacts(path: Path) -> None:
    path.mkdir()
    pd.DataFrame(
        [
            {
                "run_id": "run",
                "checkpoint_id": "run::1",
                "checkpoint_step": 1,
                "max_ledger_step_used": 1,
                "max_observation_step_used": 1,
                "is_terminal_checkpoint": False,
            }
        ]
    ).to_parquet(path / "checkpoints.parquet", index=False)
    for artifact in ESTIMATOR_ARTIFACTS:
        target = path / artifact
        if not target.exists():
            if target.suffix == ".parquet":
                pd.DataFrame({"run_id": ["run"], "checkpoint_id": ["run::1"]}).to_parquet(target, index=False)
            else:
                target.write_text("{}\n", encoding="utf-8")
