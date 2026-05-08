from __future__ import annotations

from coding_data_collection.ledger import transcript_to_wire_events


def test_transcript_to_wire_events_uses_ledger_wire_schema() -> None:
    transcript = [
        {"step": 1, "kind": "shell", "command": "pytest -q", "ts": "2026-05-05T00:00:00Z"},
        {"step": 2, "kind": "edit_file", "path": "app.py", "summary": "edit app"},
    ]
    events = transcript_to_wire_events(transcript, run_id="r1")
    assert events[0]["schema_version"].startswith("1.")
    assert events[0]["ledger_ops"][0]["category"] == "validation"
    assert events[1]["ledger_ops"][0]["category"] == "product"

