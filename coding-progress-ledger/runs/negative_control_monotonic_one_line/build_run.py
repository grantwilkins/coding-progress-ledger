from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
LEDGER_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(LEDGER_ROOT))

from ledger_progress import EventType, LedgerSession  # noqa: E402


def largest_drop(path: Path) -> float:
    rows = list(csv.DictReader(path.open()))
    values = [float(row["progress"]) for row in rows]
    return max([a - b for a, b in zip(values, values[1:]) if b < a] or [0.0])


session = LedgerSession("Negative control: monotonic one-line is_even fix")
s1 = session.add("Create minimal predicate repo with inverted parity bug", step=1)
s2 = session.add("Add hand-checkable parity regression test", step=1)
s3 = session.add("Patch is_even to test divisibility by two", step=1)
s4 = session.add("Run pytest and capture passing validation", step=1)

session.complete(s1, "repo/predicates.py and tests directory created; initial code returned odd predicate", step=2)
session.complete(s2, "tests/test_predicates.py covers even, odd, zero, and negative integers", step=3)
session.complete(s3, "final_diff.patch shows one-line change from == 1 to == 0", step=4)
session.complete(s4, "test_output.txt shows 1 passed", step=5)

session.export_jsonl(str(RUN_DIR / "ledger.jsonl"))
session.export_curve_csv(str(RUN_DIR / "progress.csv"))

events = session.ledger.events
drop = largest_drop(RUN_DIR / "progress.csv")
summary = {
    "task_id": "negative_control_monotonic_one_line",
    "final_progress": session.score().progress,
    "subtasks_created": len(session.ledger.subtasks),
    "completed_subtasks": session.score().complete_leaf_count,
    "splits": sum(1 for event in events if event.event_type == EventType.SPLIT_SUBTASK),
    "reopens": sum(1 for event in events if event.event_type == EventType.REOPEN_SUBTASK),
    "invalidations": sum(1 for event in events if event.event_type == EventType.INVALIDATE_SUBTASK),
    "largest_progress_drop": drop,
    "non_monotonic": drop > 0,
    "test_command": "../../../.venv/bin/python -m pytest -q",
    "test_status": "passed",
    "negative_control_type": "trivial_monotonic",
}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
