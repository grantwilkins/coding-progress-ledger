from __future__ import annotations

import csv
import json
from pathlib import Path

from ledger_progress import EventType, LedgerSession


RUN_ROOT = Path(__file__).resolve().parent


def progress_values(path: Path) -> list[float]:
    with path.open(newline="") as file:
        return [float(row["progress"]) for row in csv.DictReader(file)]


def main() -> None:
    session = LedgerSession("TASK 8: Package import failure")

    s1 = session.add("Create intentionally buggy package fixture", step=1)
    s2 = session.add("Add deterministic import and execution tests", step=2)
    s3 = session.add("Fix package internal imports", step=3)
    s4 = session.add("Verify required command surfaces", step=4)
    s5 = session.add("Export required run artifacts", step=5)

    session.start(s1, step=6, evidence="Created repo/ package skeleton")
    session.complete(
        s1,
        evidence="Initial git commit bf73087 contains bare internal import from helpers",
        step=7,
    )

    session.start(s2, step=8)
    children = session.split(
        s2,
        [
            "Assert python -m widget_runner.module behavior",
            "Assert direct test invocation behavior",
            "Assert package import behavior",
        ],
        step=9,
        reason="Test surface became three separately checkable commands",
    )
    session.complete(
        children[0],
        "tests/test_imports.py checks module execution subprocess output",
        step=10,
    )
    session.complete(
        children[1],
        "tests/test_imports.py is runnable directly with unittest.main()",
        step=11,
    )
    session.complete(
        children[2],
        "tests/test_imports.py imports widget_runner.build_message",
        step=12,
    )

    session.start(s3, step=13)
    session.complete(
        s3,
        "widget_runner/module.py uses from .helpers import normalize_name",
        step=14,
    )

    script_compat = session.add(
        "Preserve old script-from-package-directory execution style",
        step=15,
        parent_id=s3,
        reason="Relative import fix revealed a possible compatibility question",
    )
    session.start(
        script_compat,
        step=16,
        evidence="python widget_runner/module.py would not be a supported package execution path",
    )
    session.reopen(
        s3,
        step=17,
        reason="Import fix looked complete, but compatibility scope needed a decision",
    )
    session.invalidate(
        script_compat,
        step=18,
        reason="Out of scope; required surfaces are python -m, direct tests, and package import",
    )
    session.complete(
        s3,
        "Compatibility decision documented; package-relative import remains final fix",
        step=19,
    )

    session.start(s4, step=20)
    session.complete(
        s4,
        [
            "python3 -m widget_runner.module succeeded",
            "python3 tests/test_imports.py ran 2 tests OK",
            "python3 -c package import command succeeded",
        ],
        step=21,
    )

    session.start(s5, step=22)
    session.complete(
        s5,
        "Run root contains task docs, README, transcript, diff, test output, notes, summary, ledger, and progress CSV",
        step=23,
    )

    ledger_path = RUN_ROOT / "ledger.jsonl"
    progress_path = RUN_ROOT / "progress.csv"
    session.export_jsonl(str(ledger_path))
    session.export_curve_csv(str(progress_path))

    values = progress_values(progress_path)
    drops = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    largest_drop = max([drop for drop in drops if drop > 0], default=0.0)
    final_score = session.score()
    summary = {
        "task_id": "task_8_package_import_failure",
        "final_progress": final_score.progress,
        "subtasks_created": len(session.ledger.subtasks),
        "completed_subtasks": final_score.complete_leaf_count,
        "splits": sum(event.event_type == EventType.SPLIT_SUBTASK for event in session.ledger.events),
        "reopens": sum(event.event_type == EventType.REOPEN_SUBTASK for event in session.ledger.events),
        "invalidations": sum(event.event_type == EventType.INVALIDATE_SUBTASK for event in session.ledger.events),
        "largest_progress_drop": largest_drop,
        "non_monotonic": largest_drop > 0,
        "test_command": "python3 -m widget_runner.module && python3 tests/test_imports.py && python3 -c \"from widget_runner import build_message; print(build_message(' ada   lovelace '))\"",
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
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
