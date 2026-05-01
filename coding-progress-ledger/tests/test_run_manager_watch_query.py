from __future__ import annotations

import json
import threading
import time
from io import StringIO

import pytest

from ledger_progress import LedgerSession, SubtaskCategory
from ledger_progress.run_manager import main as run_manager_main


def _seed_run(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    session = LedgerSession("Watch test")
    sid = session.add("Patch behavior", step=1, category=SubtaskCategory.PRODUCT)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    return session, sid


def test_watch_emits_one_update_per_appended_event(tmp_path, capsys):
    run_dir = tmp_path / "run_watch"
    session, sid = _seed_run(run_dir)
    appender_done = threading.Event()

    def append_more():
        time.sleep(0.1)
        for step in range(2, 7):
            session.add(f"Subtask {step}", step=step, category=SubtaskCategory.VALIDATION)
            session.export_jsonl(str(run_dir / "ledger.jsonl"))
            time.sleep(0.05)
        appender_done.set()

    thread = threading.Thread(target=append_more)
    thread.start()
    rc = run_manager_main(["watch", str(run_dir), "--poll-interval", "0.05", "--exit-after-events", "6"])
    thread.join(timeout=5)
    assert appender_done.is_set()
    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(out_lines) == 6
    parsed = [json.loads(line) for line in out_lines]
    assert [row["event_index"] for row in parsed] == [0, 1, 2, 3, 4, 5]
    for row in parsed:
        assert "coding_progress" in row
        assert "stalled_for_blocked" in row


def test_query_status_blocked_returns_active_blocked_leaves(tmp_path, capsys):
    run_dir = tmp_path / "run_query"
    run_dir.mkdir()
    session = LedgerSession("Query test")
    a = session.add("Coding A", step=1, category=SubtaskCategory.PRODUCT)
    b = session.add("Coding B", step=2, category=SubtaskCategory.PRODUCT)
    session.block(a, step=3, reason="waiting on env")
    session.complete(b, "diff applied", step=4)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))

    rc = run_manager_main(["query", str(run_dir), "--status", "blocked"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    blocked = payload["active_blocked_leaves"]
    assert [s["id"] for s in blocked] == [a]


def test_query_stalled_reopens_discovered_validation(tmp_path, capsys):
    run_dir = tmp_path / "run_query2"
    run_dir.mkdir()
    session = LedgerSession("Query test 2")
    a = session.add("Coding", step=1, category=SubtaskCategory.PRODUCT)
    val = session.add("Validate", step=2, category=SubtaskCategory.VALIDATION)
    c = session.add("Refactor", step=2, category=SubtaskCategory.PRODUCT)
    session.block(a, step=3, reason="stuck")
    session.complete(val, "test passed", step=5)
    session.complete(c, "diff applied", step=6)
    session.reopen(c, step=7, reason="regression")
    session.add("Discovered later", step=9, category=SubtaskCategory.PRODUCT)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))

    assert run_manager_main([
        "query", str(run_dir),
        "--stalled-for", "2",
        "--reopens-since", "3",
        "--newly-discovered-since", "5",
        "--last-validation-event",
    ]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["current_step"] == 9
    assert payload["meets_threshold"] is True
    assert payload["stalled_for_threshold"] == 2
    assert any(e["event_type"] == "reopen_subtask" for e in payload["reopens_since"])
    assert [s["description"] for s in payload["newly_discovered_since"]] == ["Discovered later"]
    assert payload["last_validation_event"] is not None
    assert payload["last_validation_event"]["event_type"] in {"add_subtask", "update_status", "add_evidence"}


def test_query_requires_ledger(tmp_path):
    run_dir = tmp_path / "no_ledger"
    run_dir.mkdir()
    assert run_manager_main(["query", str(run_dir)]) == 1
