from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
LEDGER_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(LEDGER_ROOT))

from ledger_progress import EventType, LedgerSession  # noqa: E402


def largest_drop(progress_path: Path) -> float:
    rows = list(csv.DictReader(progress_path.open()))
    values = [float(row["progress"]) for row in rows]
    drops = [a - b for a, b in zip(values, values[1:]) if b < a]
    return max(drops, default=0.0)


session = LedgerSession("Task 4: CSV aggregation with messy input")
s1 = session.add("Add clean repeated-user CSV regression", step=1)
s2 = session.add("Patch aggregation to sum repeated user rows", step=1)
s3 = session.add("Compare sample input and expected output against implementation", step=1)
s4 = session.add("Validate CSV aggregator behavior", step=1)

session.complete(s1, "tests/test_aggregator.py includes clean repeated-user case", step=2)
session.complete(s2, "aggregator.py uses defaultdict totals by user_id", step=3)
session.complete(s3, "repo/data/input.csv and expected_output.csv document hand-checkable totals", step=4)

s5 = session.add("Normalize whitespace around user IDs before grouping", step=5, reason="Messy sample and tests showed ' alice ' must merge with 'alice'")
s6 = session.add("Treat blank amount cells as zero instead of crashing", step=5, reason="Messy rows can contain an empty amount")
s7 = session.add("Emit deterministic user order independent of input order", step=5, reason="Out-of-order rows otherwise make output unstable")
children = session.split(
    s4,
    [
        "Run clean-row regression after changes",
        "Run messy-row regression for whitespace and blanks",
        "Run row-order determinism regression",
    ],
    step=6,
    reason="Validation became several independent checks after messy input was discovered",
)
session.complete(s5, "aggregator.py strips row['user_id'] before grouping", step=7)
session.complete(s6, "aggregator.py maps blank amount text to 0.0", step=8)
session.complete(s7, "aggregator.py writes sorted(totals.items())", step=9)
session.complete(children[0], "test_clean_rows_sum_repeated_users passed in test_output.txt", step=10)
session.complete(children[1], "test_messy_rows_normalize_missing_values_and_sort_users passed in test_output.txt", step=10)
session.complete(children[2], "test_row_order_does_not_change_output_order_or_totals passed in test_output.txt", step=10)

session.export_jsonl(str(RUN_DIR / "ledger.jsonl"))
session.export_curve_csv(str(RUN_DIR / "progress.csv"))

events = session.ledger.events
summary = {
    "task_id": "task_4_csv_messy_aggregation",
    "final_progress": session.score().progress,
    "subtasks_created": len(session.ledger.subtasks),
    "completed_subtasks": session.score().complete_leaf_count,
    "splits": sum(1 for event in events if event.event_type == EventType.SPLIT_SUBTASK),
    "reopens": sum(1 for event in events if event.event_type == EventType.REOPEN_SUBTASK),
    "invalidations": sum(1 for event in events if event.event_type == EventType.INVALIDATE_SUBTASK),
    "largest_progress_drop": largest_drop(RUN_DIR / "progress.csv"),
    "non_monotonic": largest_drop(RUN_DIR / "progress.csv") > 0,
    "test_command": "../../../.venv/bin/python -m pytest -q",
    "test_status": "passed",
    "artifact_paths": [
        "task.md",
        "README.md",
        "agent_transcript.md",
        "ledger.jsonl",
        "progress.csv",
        "final_diff.patch",
        "test_output.txt",
        "run_notes.md",
        "summary.json",
    ],
}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
