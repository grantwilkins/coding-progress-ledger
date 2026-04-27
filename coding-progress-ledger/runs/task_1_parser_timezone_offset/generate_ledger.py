import csv
import json
from pathlib import Path

from ledger_progress import LedgerSession


RUN_ROOT = Path(__file__).resolve().parent


session = LedgerSession("TASK 1: Parser timezone offset bug")

s1 = session.add("Create tiny self-contained parser repo", step=1, reason="Initial decomposition")
s2 = session.add("Write deterministic parser tests", step=1, reason="Initial decomposition")
s3 = session.add("Confirm baseline compact-offset failure", step=1, reason="Initial decomposition")
s4 = session.add("Patch parse_offset implementation", step=1, reason="Initial decomposition")
s5 = session.add("Export required run artifacts", step=1, reason="Initial decomposition")

session.start(s1, step=2, evidence="Created repo skeleton under runs/task_1_parser_timezone_offset/repo")
session.complete(s1, "pyproject, package, parser module, and tests directory created", step=3)

session.start(s2, step=4, evidence="Added pytest cases for colon, compact positive, compact negative, and invalid inputs")
session.complete(s2, "tests/test_parser.py includes claim and plausible wrong implementations docstring", step=5)

session.start(s3, step=6, evidence="Ran baseline tests against intentionally colon-only parser")
session.complete(s3, "Baseline pytest result: 2 failed, 7 passed; +0530 and -0330 rejected", step=7)

s6 = session.add(
    "Preserve colon behavior while adding compact syntax",
    step=8,
    reason="Completing compact support must not regress the already-working +05:30 behavior",
)

session.start(s4, step=9, evidence="Changed parser regex to make the colon separator optional")
s4_children = session.split(
    s4,
    ["Accept compact positive HHMM offsets", "Accept compact negative HHMM offsets with sign"],
    step=10,
    reason="Positive compact parsing and negative sign preservation are separate likely failure modes",
)
session.complete(s4_children[0], "+0530 test now exercises 5 hours and 30 minutes grouping", step=11)
session.complete(s4_children[1], "-0330 test now exercises negative sign preservation", step=12)
session.complete(s4, "Both parser implementation child tasks completed", step=12)

session.complete(s6, "Final tests still pass for +05:30 after compact support was added", step=13)

session.complete(s5, "Created task.md, README.md, transcript, diff, test output, notes, summary, ledger, and progress CSV", step=14)
session.reopen(s5, step=15, reason="summary.json still needed computed progress-drop metrics")
session.complete(s5, "summary.json generated from ledger events and progress.csv", step=16)

ledger_path = RUN_ROOT / "ledger.jsonl"
progress_path = RUN_ROOT / "progress.csv"
session.export_jsonl(str(ledger_path))
session.export_curve_csv(str(progress_path))

with progress_path.open() as file:
    rows = list(csv.DictReader(file))

progress_values = [float(row["progress"]) for row in rows]
drops = [
    progress_values[i - 1] - progress_values[i]
    for i in range(1, len(progress_values))
    if progress_values[i] < progress_values[i - 1]
]

event_counts = {"splits": 0, "reopens": 0, "invalidations": 0}
for event in session.ledger.events:
    if event.event_type.value == "split_subtask":
        event_counts["splits"] += 1
    elif event.event_type.value == "reopen_subtask":
        event_counts["reopens"] += 1
    elif event.event_type.value == "invalidate_subtask":
        event_counts["invalidations"] += 1

active_subtasks = [
    subtask for subtask in session.ledger.subtasks.values()
    if subtask.status.value not in {"invalidated", "deleted"}
]
completed_subtasks = [
    subtask for subtask in active_subtasks
    if subtask.status.value == "complete"
]

artifact_paths = [
    "task.md",
    "README.md",
    "agent_transcript.md",
    "ledger.jsonl",
    "progress.csv",
    "final_diff.patch",
    "test_output.txt",
    "run_notes.md",
    "summary.json",
]

summary = {
    "task_id": "task_1_parser_timezone_offset",
    "final_progress": session.score().progress,
    "subtasks_created": len(session.ledger.subtasks),
    "completed_subtasks": len(completed_subtasks),
    "splits": event_counts["splits"],
    "reopens": event_counts["reopens"],
    "invalidations": event_counts["invalidations"],
    "largest_progress_drop": max(drops) if drops else 0.0,
    "non_monotonic": bool(drops),
    "test_command": "/Users/grantwilkins/houdini/coding-progress-ledger/.venv/bin/python -m pytest -q",
    "test_status": "passed",
    "artifact_paths": artifact_paths,
}

(RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
