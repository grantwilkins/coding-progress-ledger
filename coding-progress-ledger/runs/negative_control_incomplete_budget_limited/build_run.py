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


session = LedgerSession("Negative control: incomplete discount fix")
s1 = session.add("Create minimal discount repo with percent arithmetic bug", step=1)
s2 = session.add("Add hand-checkable discount regression tests", step=1)
s3 = session.add("Patch basic percent arithmetic for common case", step=1)
s4 = session.add("Run pytest and capture validation result", step=1)

session.complete(s1, "repo/discounts.py subtracts percent_off as cents in the baseline", step=2)
session.complete(s2, "tests/test_discounts.py covers 25 percent, 10 percent rounding, and full discount", step=3)
session.complete(s3, "final_diff.patch shows partial formula for non-full discounts", step=4)
s5 = session.add("Round fractional discounted cents according to test contract", step=5, reason="Pytest still fails the 999 cents at 10 percent case")
s6 = session.add("Clamp 100 percent discount to zero", step=5, reason="Pytest still fails the full-discount case")
session.complete(s4, "test_output.txt captured failing pytest output with two remaining failures", step=6)

session.export_jsonl(str(RUN_DIR / "ledger.jsonl"))
session.export_curve_csv(str(RUN_DIR / "progress.csv"))

events = session.ledger.events
drop = largest_drop(RUN_DIR / "progress.csv")
summary = {
    "task_id": "negative_control_incomplete_budget_limited",
    "final_progress": session.score().progress,
    "subtasks_created": len(session.ledger.subtasks),
    "completed_subtasks": session.score().complete_leaf_count,
    "splits": sum(1 for event in events if event.event_type == EventType.SPLIT_SUBTASK),
    "reopens": sum(1 for event in events if event.event_type == EventType.REOPEN_SUBTASK),
    "invalidations": sum(1 for event in events if event.event_type == EventType.INVALIDATE_SUBTASK),
    "largest_progress_drop": drop,
    "non_monotonic": drop > 0,
    "test_command": "../../../.venv/bin/python -m pytest -q",
    "test_status": "failed",
    "negative_control_type": "incomplete_budget_limited",
}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
