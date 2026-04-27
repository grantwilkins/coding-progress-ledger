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


session = LedgerSession("Task 6: Async stale-result bug")
s1 = session.add("Add deterministic out-of-order completion test", step=1)
s2 = session.add("Preserve newest result when an older request finishes later", step=1)
s3 = session.add("Keep loading state tied to the newest pending request", step=1)
s4 = session.add("Run async controller validation", step=1)

session.complete(s1, "tests/test_async_result.py uses ControlledFetcher release events", step=2)
session.complete(s3, "test_old_completion_does_not_clear_loading_for_newer_pending_request exposed the loading contract", step=3)
s5 = session.add("Track request identity so stale successes are ignored", step=4, reason="A loading-only fix would still allow old result overwrite")
session.reopen(s3, step=4, reason="The first validation showed loading and result freshness must be checked together")
children = session.split(
    s4,
    [
        "Validate newest result after out-of-order completions",
        "Validate old completion does not clear newer loading state",
        "Validate tests run without pytest-asyncio plugin",
    ],
    step=5,
    reason="Validation became result, loading, and self-contained execution checks",
)
s6 = session.add("Replace pytest-asyncio marker with standard asyncio.run wrappers", step=6, reason="Pytest reported async functions are not natively supported")
session.complete(s5, "async_result.py increments _latest_request_id and applies state only for the latest request", step=7)
session.complete(s6, "tests/test_async_result.py no longer imports pytest or uses pytest.mark.asyncio", step=8)
session.complete(s2, "test_out_of_order_completion_keeps_newest_result passed in test_output.txt", step=9)
session.complete(s3, "test_old_completion_does_not_clear_loading_for_newer_pending_request passed in test_output.txt", step=9)
session.complete(children[0], "Final pytest output shows result freshness test passed", step=10)
session.complete(children[1], "Final pytest output shows loading-state test passed", step=10)
session.complete(children[2], "Final pytest output from plain pytest run shows 2 passed", step=10)

session.export_jsonl(str(RUN_DIR / "ledger.jsonl"))
session.export_curve_csv(str(RUN_DIR / "progress.csv"))

events = session.ledger.events
summary = {
    "task_id": "task_6_async_stale_result",
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
