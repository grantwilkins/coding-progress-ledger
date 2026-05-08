"""
Claim:
The data-collection repo delegates progress replay/scoring to the sibling
coding-progress-ledger sidecar. Wire events must preserve explicit ledger_ops
when present, fall back to conservative transcript-derived ops otherwise, and
the generated sidecar artifact surface must be auditable without recomputing
ledger scoring locally.

Plausible wrong implementations:
- Ignore explicit ledger_ops and overwrite hand-authored semantic work events.
- Complete every inferred transcript row immediately, creating fake perfect progress.
- Produce only events.jsonl and placeholder progress artifacts without a
  sidecar-generated summary.
- Accept a summary whose source_ledger_sha256 does not match ledger.jsonl.
- Treat progress_by_category.csv as present without checking that it contains
  the coding progress surface expected by downstream consumers.
- Accept orphan sidecar outputs that are not tied to collection control
  manifests.
"""

import hashlib
from pathlib import Path

from coding_data_collection.artifacts import write_json
from coding_data_collection.ledger import transcript_to_wire_events, write_wire_events
from coding_data_collection.ledger_sidecar_audit import ledger_sidecar_report


def test_transcript_to_wire_events_preserves_explicit_ledger_ops() -> None:
    transcript = [
        {
            "step": 3,
            "kind": "shell",
            "summary": "semantic implementation",
            "ledger_ops": [
                {
                    "op": "add",
                    "id": "product_parser",
                    "description": "Implement parser",
                    "category": "product",
                },
                {
                    "op": "complete",
                    "id": "product_parser",
                    "evidence": "Parser handles quoted fields",
                },
            ],
        }
    ]

    events = transcript_to_wire_events(transcript, run_id="r1")

    assert events[0]["ledger_ops"] == transcript[0]["ledger_ops"]


def test_transcript_to_wire_events_falls_back_to_transcript_derived_ops() -> None:
    events = transcript_to_wire_events(
        [{"step": 1, "kind": "shell", "command": "pytest -q", "summary": "run tests"}],
        run_id="r1",
    )

    assert events[0]["ledger_ops"][0]["category"] == "validation"
    assert events[0]["ledger_ops"][1]["op"] == "start"


def test_transcript_to_wire_events_only_completes_inferred_done_boundary() -> None:
    events = transcript_to_wire_events(
        [
            {"step": 1, "kind": "shell", "summary": "investigate"},
            {"step": 2, "kind": "done", "summary": "agent done"},
        ],
        run_id="r1",
    )

    assert events[0]["ledger_ops"][1]["op"] == "start"
    assert events[1]["ledger_ops"][1]["op"] == "complete"


def test_transcript_to_wire_events_complete_concrete_successful_tool_rows() -> None:
    events = transcript_to_wire_events(
        [
            {"step": 1, "kind": "read_file", "path": "task.md", "exit_code": 0, "summary": "read task"},
            {"step": 2, "kind": "write_file", "path": "answer.txt", "exit_code": 0, "summary": "write answer"},
            {"step": 3, "kind": "shell", "command": "pytest -q", "exit_code": 0, "summary": "run tests"},
        ],
        run_id="r1",
    )

    assert [event["ledger_ops"][1]["op"] for event in events] == ["complete", "complete", "complete"]
    assert events[1]["ledger_ops"][0]["category"] == "product"
    assert events[2]["ledger_ops"][0]["category"] == "validation"


def test_transcript_to_wire_events_reopen_visible_work_on_verifier_failure() -> None:
    events = transcript_to_wire_events(
        [
            {"step": 1, "kind": "write_file", "path": "answer.txt", "exit_code": 0, "summary": "write answer"},
            {"step": 2, "kind": "shell", "command": "python -m py_compile answer.py", "exit_code": 0},
        ],
        run_id="r1",
        verifier_exit_code=1,
    )

    terminal = events[-1]
    assert terminal["step"] == 3
    assert terminal["agent_step"]["visible_to_agent"] is False
    assert [op["op"] for op in terminal["ledger_ops"]] == ["add", "block", "reopen"]
    assert terminal["ledger_ops"][-1]["id"] == events[1]["ledger_ops"][1]["id"]


def test_transcript_to_wire_events_do_not_create_terminal_drop_on_success() -> None:
    events = transcript_to_wire_events(
        [{"step": 1, "kind": "write_file", "path": "answer.txt", "exit_code": 0, "summary": "write answer"}],
        run_id="r1",
        verifier_exit_code=0,
    )

    terminal = events[-1]
    assert terminal["step"] == 2
    assert terminal["ledger_ops"] == []


def test_ledger_sidecar_report_requires_generated_artifacts_and_matching_summary_hash(tmp_path: Path) -> None:
    transcript = [{"step": 1, "kind": "shell", "command": "pytest -q", "summary": "run tests"}]
    write_wire_events(tmp_path / "events.jsonl", transcript_to_wire_events(transcript, run_id="r1"))
    (tmp_path / "ledger.jsonl").write_text('{"event_type":"init"}\n', encoding="utf-8")
    (tmp_path / "progress.csv").write_text("step,progress\n0,0.0\n", encoding="utf-8")
    (tmp_path / "progress_by_category.csv").write_text("step,coding_progress\n0,0.0\n", encoding="utf-8")
    write_json(
        tmp_path / "summary_by_category.json",
        {
            "generator": "ledger_progress.sidecar",
            "source_ledger_sha256": "not-the-ledger-hash",
        },
    )

    report = ledger_sidecar_report(tmp_path)

    assert report["passed"] is False
    assert "summary_by_category.json: source_ledger_sha256 does not match ledger.jsonl" in report["issues"]


def test_ledger_sidecar_report_requires_collection_control_artifacts(tmp_path: Path) -> None:
    transcript = [{"step": 1, "kind": "shell", "command": "pytest -q", "summary": "run tests"}]
    write_wire_events(tmp_path / "events.jsonl", transcript_to_wire_events(transcript, run_id="r1"))
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text('{"event_type":"init"}\n', encoding="utf-8")
    (tmp_path / "progress.csv").write_text("step,progress\n0,0.0\n", encoding="utf-8")
    (tmp_path / "progress_by_category.csv").write_text("step,coding_progress\n0,0.0\n", encoding="utf-8")
    write_json(
        tmp_path / "summary_by_category.json",
        {
            "generator": "ledger_progress.sidecar",
            "source_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        },
    )

    report = ledger_sidecar_report(tmp_path)

    assert report["passed"] is False
    assert report["collection_run_valid"] is False
    assert "run_manifest.json" in report["missing_collection_control_artifacts"]
    assert "run_manifest.json: missing collection control artifact" in report["issues"]
