"""
Claim:
The live sidecar converts wire-format JSONL into a deterministic ledger
measurement trace: explicit ops are authoritative, adapter inference records
visible attempted/completed work without predicting success, timestamps remain
event-local, and one run directory contains one run_id.

Plausible wrong implementations:
- Reuse the first timestamp or wall clock for later ledger events.
- Treat any command, submit provenance, or final exit_status as completed work.
- Ignore explicit wire ids/categories/weights and let LedgerSession allocate them.
- Merge multiple run_id streams into one ledger.
- Reject additive v1.x wire-format events even though the protocol allows them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from ledger_progress import EventType, Status, SubtaskCategory, from_jsonl, score
from ledger_progress.adapters import swe_agent
from ledger_progress.run_manager import main as run_manager_main
from ledger_progress.sidecar import LedgerSidecar


TS = "2026-04-30T12:00:00+00:00"


def test_sidecar_consumes_synthetic_stream_and_writes_valid_run_dir(tmp_path):
    run_dir = tmp_path / "live_run"
    sidecar = LedgerSidecar(run_dir, "generic")

    sidecar.process_lines(json.dumps(_event(i, agent_step={"tool_name": f"tool{i}", "command": f"tool{i} arg"})) + "\n" for i in range(1, 6))

    ledger = from_jsonl(str(run_dir / "ledger.jsonl"))
    assert len(ledger.events) == 11
    assert all(event.timestamp is not None for event in ledger.events)
    assert (run_dir / "progress.csv").exists()
    assert (run_dir / "progress_by_category.csv").exists()
    assert (run_dir / "summary_by_category.json").exists()
    assert run_manager_main(["check-run", str(run_dir)]) == 0


def test_schema_major_version_mismatch_raises(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        sidecar.process_event(_event(1, schema_version="2.0", agent_step={"tool_name": "edit"}))


def test_additive_minor_version_and_unknown_fields_do_not_change_measurement(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")

    sidecar.process_event(_event(1, schema_version="1.1", agent_step={"tool_name": "edit"}, extra_field={"ignored": True}))

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert set(ledger.subtasks) == {"G1"}
    assert score(ledger).progress == 1.0


def test_replay_equality_for_same_input(tmp_path):
    lines = [json.dumps(_event(i, agent_step={"tool_name": "edit", "command": "edit 1:1 x", "observation": "ok"})) + "\n" for i in range(1, 4)]

    LedgerSidecar(tmp_path / "run_a", "swe_agent").process_lines(lines)
    LedgerSidecar(tmp_path / "run_b", "swe_agent").process_lines(lines)

    assert (tmp_path / "run_a" / "ledger.jsonl").read_bytes() == (tmp_path / "run_b" / "ledger.jsonl").read_bytes()


def test_timestamp_authority_survives_exactly_per_input_event(tmp_path):
    first_timestamp = "2026-04-30T12:34:56.123456+00:00"
    second_timestamp = "2026-04-30T12:35:10.654321+00:00"
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")

    sidecar.process_event(_event(7, timestamp=first_timestamp, ledger_ops=[{"op": "add", "id": "S1", "description": "Patch code"}]))
    sidecar.process_event(_event(8, timestamp=second_timestamp, ledger_ops=[{"op": "complete", "id": "S1", "evidence": ["step 8: patch applied"]}]))

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert [event.timestamp for event in ledger.events] == [first_timestamp, first_timestamp, second_timestamp]


def test_explicit_ops_bypass_inferrer(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "swe_agent")
    sidecar.process_event(
        _event(
            1,
            agent_step={"tool_name": "edit", "command": "edit 1:1 x", "observation": "ok"},
            ledger_ops=[
                {"op": "add", "id": "E1", "category": "validation", "description": "Explicit validation"},
                {"op": "complete", "id": "E1", "evidence": ["step 1: explicit evidence"]},
            ],
        )
    )

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert set(ledger.subtasks) == {"E1"}
    assert ledger.subtasks["E1"].category is SubtaskCategory.VALIDATION


def test_explicit_op_dispatch_covers_scope_changes_and_evidence(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")
    sidecar.process_event(_event(1, ledger_ops=[{"op": "add", "id": "S1", "description": "Broad task"}]))
    sidecar.process_event(
        _event(
            2,
            ledger_ops=[
                {
                    "op": "split",
                    "id": "S1",
                    "reason": "Need checkable leaves",
                    "children": [
                        {"id": "S1.1", "description": "Patch behavior", "category": "product"},
                        {"id": "S1.2", "description": "Run validation", "category": "validation"},
                    ],
                }
            ],
        )
    )
    sidecar.process_event(_event(3, ledger_ops=[{"op": "add_evidence", "id": "S1.1", "evidence": ["step 3: diff changed code"]}]))
    sidecar.process_event(_event(4, ledger_ops=[{"op": "complete", "id": "S1.1", "evidence": ["step 4: patch applied"]}]))
    sidecar.process_event(_event(5, ledger_ops=[{"op": "reopen", "id": "S1.1", "reason": "Review found missing edge case"}]))
    sidecar.process_event(_event(6, ledger_ops=[{"op": "block", "id": "S1.2", "reason": "Tests unavailable", "evidence": ["step 6: pytest missing"]}]))
    sidecar.process_event(_event(7, ledger_ops=[{"op": "invalidate", "id": "S1.2", "reason": "Validation path replaced"}]))

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert ledger.subtasks["S1.1"].status is Status.IN_PROGRESS
    assert ledger.subtasks["S1.2"].status is Status.INVALIDATED
    assert ledger.subtasks["S1.1"].category is SubtaskCategory.PRODUCT
    assert ledger.subtasks["S1.2"].category is SubtaskCategory.VALIDATION
    assert EventType.ADD_EVIDENCE in [event.event_type for event in ledger.events]
    assert EventType.SPLIT_SUBTASK in [event.event_type for event in ledger.events]
    assert EventType.REOPEN_SUBTASK in [event.event_type for event in ledger.events]
    assert EventType.INVALIDATE_SUBTASK in [event.event_type for event in ledger.events]


def test_explicit_add_preserves_wire_id_category_parent_and_weight(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")

    sidecar.process_event(_event(1, ledger_ops=[{"op": "add", "id": "P", "description": "Parent", "category": "investigation", "weight": 3.0}]))
    sidecar.process_event(_event(2, ledger_ops=[{"op": "add", "id": "P.child", "parent_id": "P", "description": "Child", "category": "validation", "weight": 2.5}]))

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert list(ledger.subtasks) == ["P", "P.child"]
    assert ledger.subtasks["P"].category is SubtaskCategory.INVESTIGATION
    assert ledger.subtasks["P"].weight == 3.0
    assert ledger.subtasks["P.child"].parent_id == "P"
    assert ledger.subtasks["P.child"].category is SubtaskCategory.VALIDATION
    assert ledger.subtasks["P.child"].weight == 2.5


def test_sidecar_rejects_mixed_run_ids_before_mutating_ledger(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")
    sidecar.process_event(_event(1, ledger_ops=[{"op": "add", "id": "S1", "description": "Patch code"}]))

    before = (tmp_path / "live_run" / "ledger.jsonl").read_bytes()
    with pytest.raises(ValueError, match="one run_id"):
        sidecar.process_event(_event(2, run_id="other_run", ledger_ops=[{"op": "add", "id": "S2", "description": "Other work"}]))

    assert (tmp_path / "live_run" / "ledger.jsonl").read_bytes() == before
    assert set(from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl")).subtasks) == {"S1"}


def test_swe_agent_adapter_maps_vocabulary_to_categories():
    cases = [
        ("search_file getip.php", "search_file", "investigation"),
        ("edit 274:274 x", "edit", "product"),
        ("pytest tests/test_core.py", "pytest", "validation"),
        ("pip install -e .", "pip", "environment"),
        ("submit", "submit", "artifact"),
    ]

    for step, (command, tool, category) in enumerate(cases, 1):
        ops = swe_agent.infer_events({"step": step, "command": command, "tool_name": tool, "observation": "ok"})
        assert ops[0]["category"] == category


def test_swe_agent_adapter_does_not_complete_work_from_command_or_exit_status_alone(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "swe_agent")
    sidecar.process_event(
        _event(
            1,
            agent_step={
                "tool_name": "pytest",
                "command": "pytest tests/test_core.py",
                "exit_status": "submitted (exit_context)",
            },
        )
    )

    ledger = from_jsonl(str(tmp_path / "live_run" / "ledger.jsonl"))
    assert ledger.subtasks["SW1"].category is SubtaskCategory.VALIDATION
    assert ledger.subtasks["SW1"].status is Status.IN_PROGRESS
    assert score(ledger).progress == 0.0


def test_cli_entrypoint_reads_input_file(tmp_path):
    input_path = tmp_path / "events.jsonl"
    input_path.write_text(json.dumps(_event(1, agent_step={"tool_name": "edit", "command": "edit 1:1 x", "observation": "ok"})) + "\n")
    run_dir = tmp_path / "live_run"

    completed = subprocess.run(
        [sys.executable, "-m", "ledger_progress.sidecar", "--run-dir", str(run_dir), "--adapter", "swe_agent", "--input-file", str(input_path)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    ledger = from_jsonl(str(run_dir / "ledger.jsonl"))
    assert ledger.subtasks["SW1"].category is SubtaskCategory.PRODUCT


def test_sidecar_latency_under_100ms_per_event(tmp_path):
    sidecar = LedgerSidecar(tmp_path / "live_run", "generic")
    events = [_event(i, agent_step={"tool_name": "edit", "command": "edit 1:1 x", "observation": "ok"}) for i in range(1, 51)]

    started = time.monotonic()
    for event in events:
        sidecar.process_event(event)
    elapsed = time.monotonic() - started

    assert elapsed / len(events) < 0.1


def _event(
    step: int,
    *,
    schema_version: str = "1.0",
    run_id: str = "live_test",
    timestamp: str = TS,
    agent_step: dict | None = None,
    ledger_ops: list[dict] | None = None,
    **extra,
) -> dict:
    data = {
        "schema_version": schema_version,
        "run_id": run_id,
        "step": step,
        "timestamp": timestamp,
        **extra,
    }
    if agent_step is not None:
        data["agent_step"] = agent_step
    if ledger_ops is not None:
        data["ledger_ops"] = ledger_ops
    return data
