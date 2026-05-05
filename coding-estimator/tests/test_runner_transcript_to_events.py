"""
Claim:
transcript_to_events converts a subagent transcript into upstream wire-
format events. Each consecutive run of same-category actions becomes
one (add, complete) op pair. The classifier mirrors upstream
auto_annotate_hermes; pip/apt → ENVIRONMENT, pytest/unittest →
VALIDATION, ls/cat/grep → INVESTIGATION, redirection / sed -i → PRODUCT.
The `done` line is a terminator; lines after are dropped.

Plausible wrong implementations:
- pytest classified as INVESTIGATION (wrong category — would mis-label
  validation events as investigation in the ledger).
- pip install classified as PRODUCT (the command does not produce
  artifacts, it changes the env; ENVIRONMENT is correct).
- Each transcript line emits its own (add, complete) pair instead of
  grouping consecutive same-category runs (would inflate leaf count
  and break upstream interop with auto_annotate_hermes).
- Lines after `done` get included (would corrupt the ledger).
- Missing required fields on a transcript line silently default
  (should hard-fail per § 0.10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_estimator.runner.transcript_to_events import (
    classify,
    read_transcript,
    transcript_to_events,
    write_events,
)


def _line(step, kind, **kw):
    return {"step": step, "ts": f"2026-05-05T00:00:{step:02d}Z", "kind": kind, **kw}


def test_pytest_is_validation_not_investigation():
    """A pytest invocation must classify as VALIDATION; if it falls
    through to INVESTIGATION the ledger labels are wrong."""
    assert classify(_line(1, "shell", command="pytest tests/")) == "validation"
    assert classify(_line(1, "shell", command="python -m unittest tests")) == "validation"


def test_pip_install_is_environment_not_product():
    """pip install changes env state; classifying it as PRODUCT would
    let arbitrary installs count as user-visible discovered work."""
    assert classify(_line(1, "shell", command="pip install requests")) == "environment"
    assert classify(_line(1, "shell", command="apt-get install -y curl")) == "environment"
    assert classify(_line(1, "shell", command="uv add httpx")) == "environment"


def test_redirection_is_product():
    assert classify(_line(1, "shell", command="echo hi > /app/out.txt")) == "product"
    assert classify(_line(1, "shell", command="sed -i 's/a/b/' /app/x.py")) == "product"


def test_ls_cat_grep_are_investigation():
    assert classify(_line(1, "shell", command="ls /app")) == "investigation"
    assert classify(_line(1, "shell", command="cat /app/x.py")) == "investigation"
    assert classify(_line(1, "shell", command="grep -r foo /app")) == "investigation"


def test_kind_drives_classification_for_file_ops():
    assert classify(_line(1, "read_file", path="/app/x.py")) == "investigation"
    assert classify(_line(1, "write_file", path="/app/x.py")) == "product"
    assert classify(_line(1, "edit_file", path="/app/x.py")) == "product"


def test_thought_is_unclassifiable_and_filtered_upstream():
    """thought lines are not actions; they get dropped before grouping.
    Calling classify on a thought line would be a programming bug and
    must hard-fail."""
    with pytest.raises(ValueError):
        classify(_line(1, "thought", summary="planning"))


def test_consecutive_same_category_groups_into_one_leaf():
    transcript = [
        _line(1, "read_file", path="/app/a.py"),
        _line(2, "read_file", path="/app/b.py"),
        _line(3, "read_file", path="/app/c.py"),
        _line(4, "done", summary="ok"),
    ]
    events = transcript_to_events(transcript, run_id="r")
    # 1 group -> 2 events (add + complete)
    assert len(events) == 2
    add, complete = events
    assert add["ledger_ops"][0]["op"] == "add"
    assert add["ledger_ops"][0]["category"] == "investigation"
    assert add["ledger_ops"][0]["id"] == "S1"
    assert add["step"] == 1
    assert complete["ledger_ops"][0]["op"] == "complete"
    assert complete["ledger_ops"][0]["id"] == "S1"
    assert complete["step"] == 3


def test_category_change_starts_new_leaf():
    transcript = [
        _line(1, "read_file", path="/app/a.py"),       # INVESTIGATION
        _line(2, "write_file", path="/app/a.py"),      # PRODUCT
        _line(3, "shell", command="pytest tests/"),    # VALIDATION
        _line(4, "done", summary="ok"),
    ]
    events = transcript_to_events(transcript, run_id="r")
    # 3 groups -> 6 events
    assert len(events) == 6
    cats = [e["ledger_ops"][0].get("category") for e in events if e["ledger_ops"][0]["op"] == "add"]
    assert cats == ["investigation", "product", "validation"]
    ids = [e["ledger_ops"][0]["id"] for e in events]
    assert ids == ["S1", "S1", "S2", "S2", "S3", "S3"]


def test_lines_after_done_are_dropped():
    transcript = [
        _line(1, "read_file", path="/app/a.py"),
        _line(2, "done", summary="ok"),
        _line(3, "write_file", path="/app/leak.py"),  # must NOT appear
    ]
    events = transcript_to_events(transcript, run_id="r")
    assert len(events) == 2  # one (add, complete) pair only
    cats = [e["ledger_ops"][0].get("category") for e in events
            if e["ledger_ops"][0]["op"] == "add"]
    assert cats == ["investigation"]


def test_empty_or_thought_only_transcript_yields_no_events():
    """Investigation that never acts shouldn't be a 0-leaf ledger."""
    assert transcript_to_events([], run_id="r") == []
    assert transcript_to_events(
        [_line(1, "thought", summary="..."), _line(2, "done", summary="gave up")],
        run_id="r",
    ) == []


def test_wire_event_shape_matches_upstream_sidecar():
    """events must carry schema_version, run_id, step, timestamp,
    ledger_ops (a list of one op). Anything else and the upstream
    sidecar's _validate_event rejects them."""
    events = transcript_to_events(
        [_line(1, "write_file", path="/app/x.py"), _line(2, "done", summary="ok")],
        run_id="run-7",
    )
    for e in events:
        assert e["schema_version"] == "1.0"
        assert e["run_id"] == "run-7"
        assert isinstance(e["step"], int)
        assert isinstance(e["timestamp"], str) and e["timestamp"].endswith("Z")
        assert isinstance(e["ledger_ops"], list) and len(e["ledger_ops"]) == 1


def test_read_transcript_hard_fails_on_missing_required_field(tmp_path):
    """§ 0.10: hard fail over silent fallback. A transcript line
    missing 'kind' must not be silently classified as INVESTIGATION."""
    p = tmp_path / "transcript.jsonl"
    p.write_text(json.dumps({"step": 1, "ts": "2026-05-05T00:00:00Z"}) + "\n")
    with pytest.raises(ValueError, match="missing required field"):
        read_transcript(p)


def test_write_events_jsonl_one_per_line(tmp_path):
    events = [
        {"schema_version": "1.0", "run_id": "r", "step": 1,
         "timestamp": "2026-05-05T00:00:00Z",
         "ledger_ops": [{"op": "add", "step": 1, "id": "S1",
                         "category": "product", "description": "x"}]},
    ]
    out = tmp_path / "events.jsonl"
    write_events(events, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == events[0]


def test_classifier_is_pure_no_side_effects(tmp_path):
    """Classifier must not reach the filesystem. (Test by passing a
    path that doesn't exist; should not raise.)"""
    classify(_line(1, "read_file", path="/does/not/exist"))
    classify(_line(1, "shell", command="ls /does/not/exist"))
