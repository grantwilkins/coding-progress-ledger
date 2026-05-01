from __future__ import annotations

import json

import pytest

from ledger_progress import SubtaskCategory, from_jsonl
from ledger_progress.run_manager import main as run_manager_main
from scripts.run_swe_agent_live_sidecar import materialize_live_run, wire_events


def test_wire_events_pair_assistant_command_with_following_tool_observation():
    normalized = _normalized()

    events = list(wire_events(normalized, "run-1", _clock()))

    assert [event["step"] for event in events] == [2, 4, 6]
    assert [event["timestamp"] for event in events] == [
        "2026-04-30T12:00:01+00:00",
        "2026-04-30T12:00:02+00:00",
        "2026-04-30T12:00:03+00:00",
    ]
    assert events[0]["agent_step"]["command"] == "search_file bug"
    assert events[0]["agent_step"]["observation"] == "found bug.py"
    assert events[0]["agent_step"]["exit_status"] is None
    assert events[-1]["agent_step"]["command"] == "submit"
    assert events[-1]["agent_step"]["exit_status"] == "submitted"


def test_materialize_live_run_streams_wire_events_through_sidecar(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "live"
    _write_source_run(source)

    materialize_live_run(source, output, clock=_clock())

    ledger = from_jsonl(str(output / "ledger.jsonl"))
    wire = [json.loads(line) for line in (output / "wire_events.jsonl").read_text().splitlines()]
    live_metadata = json.loads((output / "live_instrumentation.json").read_text())

    assert len(wire) == 3
    assert set(ledger.subtasks) == {"SW2", "SW4", "SW6"}
    assert ledger.subtasks["SW2"].category is SubtaskCategory.INVESTIGATION
    assert ledger.subtasks["SW4"].category is SubtaskCategory.PRODUCT
    assert ledger.subtasks["SW6"].category is SubtaskCategory.ARTIFACT
    assert all(event.timestamp is not None for event in ledger.events)
    assert live_metadata["mode"] == "normalized_trace_live_sidecar_replay"
    assert live_metadata["wire_event_count"] == 3
    assert run_manager_main(["check-run", str(output)]) == 0


def test_materialize_live_run_refuses_to_overwrite_existing_ledger(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "live"
    _write_source_run(source)
    output.mkdir()
    (output / "ledger.jsonl").write_text("{}\n")

    with pytest.raises(FileExistsError):
        materialize_live_run(source, output, clock=_clock())


def _write_source_run(path):
    path.mkdir()
    (path / "normalized_trace.json").write_text(json.dumps(_normalized()))
    (path / "task.md").write_text("# Task\n\nFix the bug.\n")
    (path / "final_diff.patch").write_text("diff --git a/bug.py b/bug.py\n")
    (path / "test_output.txt").write_text("passed\n")
    (path / "source_metadata.json").write_text(json.dumps({
        "final_success": True,
        "final_success_source": "source_label",
    }))


def _normalized():
    return {
        "instance_id": "repo__issue-1",
        "exit_status": "submitted",
        "final_success": True,
        "events": [
            {"step_index": 0, "role": "system"},
            {"step_index": 1, "role": "environment", "observation": "Fix the bug"},
            {
                "step_index": 2,
                "role": "assistant",
                "thought": "Find it",
                "action": "search_file bug",
                "command": "search_file bug",
                "tool_name": "search_file",
                "files_touched": [],
            },
            {"step_index": 3, "role": "tool", "observation": "found bug.py"},
            {
                "step_index": 4,
                "role": "assistant",
                "thought": "Patch it",
                "action": "edit 1:1 fixed",
                "command": "edit 1:1 fixed",
                "tool_name": "edit",
                "files_touched": ["bug.py"],
            },
            {"step_index": 5, "role": "tool", "observation": "edited"},
            {
                "step_index": 6,
                "role": "assistant",
                "thought": "Submit",
                "action": "submit",
                "command": "submit",
                "tool_name": "submit",
                "files_touched": [],
            },
        ],
    }


def _clock():
    values = iter([
        "2026-04-30T12:00:01+00:00",
        "2026-04-30T12:00:02+00:00",
        "2026-04-30T12:00:03+00:00",
    ])
    return lambda: next(values)
