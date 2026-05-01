"""Workstream I tests: lock in semantic claims of scripts/collect_schema_gaps.py
(I1 collector) and the I2 schema decision's enum invariants.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_schema_gaps import (  # noqa: E402
    extract_schema_gap_section,
    is_none_body,
    pilot_records,
)
from ledger_progress.core import EventType, Status, SubtaskCategory  # noqa: E402

RUNS = ROOT / "runs" / "swe_agent_pilot"


def test_pilot_records_one_per_pilot_dir():
    records = pilot_records(RUNS)
    pilot_dirs = sorted(RUNS.glob("swe_agent_pilot_*"))
    assert len(records) == len(pilot_dirs)
    assert [r["pilot_id"] for r in records] == [d.name for d in pilot_dirs]


def test_schema_gap_found_count_matches_quality_files():
    truth = sum(
        1
        for d in RUNS.glob("swe_agent_pilot_*")
        if json.loads((d / "annotation_quality.json").read_text())["whether_schema_gap_found"]
    )
    records = pilot_records(RUNS)
    assert truth == 2
    assert sum(1 for r in records if r["schema_gap_found"]) == truth


def test_extract_section_stops_at_next_heading(tmp_path: Path):
    notes = tmp_path / "run_notes.md"
    notes.write_text(
        "### 7. Earlier\n\nignore me.\n\n"
        "### 8. Schema gaps observed\n\n"
        "None observed.\n\n"
        "### 9. Followups\n\n"
        "DO NOT INCLUDE\n",
        encoding="utf-8",
    )
    body = extract_schema_gap_section(notes)
    assert body == "None observed."
    assert "DO NOT INCLUDE" not in body
    assert "### 9." not in body


def test_extract_section_returns_tail_when_no_next_heading(tmp_path: Path):
    notes = tmp_path / "run_notes.md"
    notes.write_text(
        "### 8. Schema gaps observed\n\nReal gap text.\n", encoding="utf-8"
    )
    assert extract_schema_gap_section(notes) == "Real gap text."


def test_extract_section_hard_fails_when_missing(tmp_path: Path):
    notes = tmp_path / "run_notes.md"
    notes.write_text("### 7. Other\n\nno gap section here.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing '### 8. Schema gaps observed'"):
        extract_schema_gap_section(notes)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("None.", True),
        ("None observed.", True),
        ("None — refined § 6 before annotating.", True),
        (
            "**One real gap, surfaced by f_02 and resolved before annotating:** "
            "the original stuck-loop rule only covered cycles of identical commands.",
            False,
        ),
    ],
)
def test_is_none_body_classification(body: str, expected: bool):
    assert is_none_body(body) is expected


def test_status_enum_has_exact_six_values():
    assert {s.value for s in Status} == {
        "not_started",
        "in_progress",
        "blocked",
        "complete",
        "invalidated",
        "deleted",
    }


def test_event_type_enum_has_exact_eight_values():
    assert {e.value for e in EventType} == {
        "init",
        "add_subtask",
        "update_status",
        "add_evidence",
        "split_subtask",
        "reopen_subtask",
        "invalidate_subtask",
        "delete_subtask",
    }


def test_subtask_category_enum_has_exact_six_values():
    assert {c.value for c in SubtaskCategory} == {
        "product",
        "validation",
        "investigation",
        "environment",
        "artifact",
        "documentation",
    }
