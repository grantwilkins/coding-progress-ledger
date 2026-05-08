"""RunRecord loader: events sorted by step, ledger_path exists,
has_real_wallclock matches live_instrumentation.json claim,
hard fails on a missing ledger.jsonl.

Claim:
    _events_sorted returns events ordered by step ascending; on ties, the
    original file-input order is preserved (stable sort).

Plausible wrong implementations:
    - switch to heapq.nsmallest -> not stable on ties
    - switch to numpy.argsort default (quicksort) -> not stable
    - sort by (step, event_type) -> would reorder tied events alphabetically
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.ingest.run_record import load_run


@pytest.fixture()
def real_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)


def test_tb_live_run_loads_with_real_wallclock(real_ledger: None) -> None:
    rec = load_run("tb_live", "markdown-to-html-cli")
    assert rec.run_id == "markdown-to-html-cli"
    assert rec.source == "tb_live"
    assert rec.ledger_path.is_file()
    assert rec.has_real_wallclock is True
    assert rec.start_wall_time is not None and rec.end_wall_time is not None
    assert rec.start_wall_time <= rec.end_wall_time
    assert rec.task_id == "markdown-to-html-cli"
    # Events are sorted by step (stable), so consecutive steps must be
    # non-decreasing.
    steps = [e.step for e in rec.events]
    assert steps == sorted(steps)
    assert len(rec.events) >= 1
    assert rec.events[0].event_type.value == "init"


def test_swe_agent_pilot_has_no_real_wallclock(real_ledger: None) -> None:
    rec = load_run("swe_agent_pilot", "swe_agent_pilot_s_06")
    assert rec.has_real_wallclock is False
    assert rec.start_wall_time is None and rec.end_wall_time is None
    assert rec.model_name == "swe-agent-llama-70b"
    assert rec.task_id == "mahmoud__boltons-298"


def test_hermes_pilot_h5_v2_loads(real_ledger: None) -> None:
    rec = load_run("hermes_pilot_h5_v2", "hermes_pilot_h5_001")
    assert rec.has_real_wallclock is False  # synthetic timestamps
    assert rec.task_family == "File Operations"
    assert rec.model_name == "glm-5.1"


def test_tb_live_v2_loads_with_manifest_wallclock_and_metadata(real_ledger: None) -> None:
    rec = load_run(
        "tb_live_v2",
        "validation_new_work_05_quoted_field_in_tsv__armB__87f7ab5e",
    )
    assert rec.has_real_wallclock is True
    assert rec.start_wall_time is not None and rec.end_wall_time is not None
    assert rec.start_wall_time <= rec.end_wall_time
    assert rec.task_id == "validation_new_work_05_quoted_field_in_tsv"
    assert rec.task_family == "validation_new_work"
    assert rec.arm == "B"
    assert rec.difficulty == "medium"
    assert rec.model_name == "claude-sonnet-4-6"
    assert rec.agent_scaffold == "general-purpose"


def test_missing_ledger_hard_fails(
    real_ledger: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path / "ledger"
    (fake_root / "ledger_progress").mkdir(parents=True)
    runs = fake_root / "runs" / "tb_live" / "no_ledger"
    runs.mkdir(parents=True)
    monkeypatch.setenv("LEDGER_ROOT", str(fake_root))
    with pytest.raises(FileNotFoundError, match="ledger.jsonl missing"):
        load_run("tb_live", "no_ledger")


def test_unknown_source_raises(real_ledger: None) -> None:
    with pytest.raises(KeyError):
        load_run("does_not_exist", "any_run")


def test_event_sort_is_stable_with_handcrafted_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five events at three distinct steps; three of them share step==2 with
    # event_types in REVERSE alphabetical order (z, m, a). A stable sort by
    # step alone preserves file order -> output is z, m, a. A sort that uses
    # (step, event_type) as the key would reorder to a, m, z. A non-stable
    # sort (heapq, numpy quicksort) might produce ANY of the 6 permutations.
    fake_root = tmp_path / "ledger"
    (fake_root / "ledger_progress").mkdir(parents=True)
    rd = fake_root / "runs" / "tb_live" / "ordered_run"
    rd.mkdir(parents=True)
    # Use upstream-valid event_type values; pick three add_subtask events
    # with subtask_ids that sort REVERSE-alphabetically in file order.
    lines = [
        '{"step":0,"event_type":"init","subtask_id":null,"payload":{},"reason":null}',
        '{"step":2,"event_type":"add_subtask","subtask_id":"z","payload":{"description":"z","parent_id":null,"weight":1.0,"category":"product"},"reason":null}',
        '{"step":2,"event_type":"add_subtask","subtask_id":"m","payload":{"description":"m","parent_id":null,"weight":1.0,"category":"product"},"reason":null}',
        '{"step":2,"event_type":"add_subtask","subtask_id":"a","payload":{"description":"a","parent_id":null,"weight":1.0,"category":"product"},"reason":null}',
        '{"step":5,"event_type":"update_status","subtask_id":"z","payload":{"status":"complete"},"reason":null}',
    ]
    (rd / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("LEDGER_ROOT", str(fake_root))
    rec = load_run("tb_live", "ordered_run")
    same_step = [e for e in rec.events if e.step == 2]
    assert [e.subtask_id for e in same_step] == ["z", "m", "a"]


def test_has_real_wallclock_false_when_live_file_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when the source registry says timestamp_quality='real',
    has_real_wallclock must be False if live_instrumentation.json doesn't
    actually claim a wallclock timestamp_source."""
    import json as _json

    fake_root = tmp_path / "ledger"
    (fake_root / "ledger_progress").mkdir(parents=True)
    rd = fake_root / "runs" / "tb_live" / "fake_run"
    rd.mkdir(parents=True)
    (rd / "ledger.jsonl").write_text(
        '{"step":0,"event_type":"init","subtask_id":null,'
        '"payload":{},"reason":null,"timestamp":"2026-05-04T00:00:00Z"}\n'
    )
    # live_instrumentation present but with a non-wallclock timestamp_source
    (rd / "live_instrumentation.json").write_text(
        _json.dumps({"timestamp_source": "step_index", "task_id": "fake"})
    )
    monkeypatch.setenv("LEDGER_ROOT", str(fake_root))
    rec = load_run("tb_live", "fake_run")
    assert rec.has_real_wallclock is False
    assert rec.start_wall_time is None and rec.end_wall_time is None


def test_event_sort_preserves_file_order_within_step(real_ledger: None) -> None:
    # In tb_live, several steps emit multiple events. Confirm stable sort
    # preserves their input file order: an add_subtask is followed by an
    # update_status at the same step in markdown-to-html-cli (step 1).
    rec = load_run("tb_live", "markdown-to-html-cli")
    pairs = [(e.step, e.event_type.value) for e in rec.events]
    # find a same-step pair
    for i in range(len(pairs) - 1):
        if pairs[i][0] == pairs[i + 1][0]:
            # the canonical order for an add_subtask + update_status pair is
            # add_subtask first (creation precedes status update).
            if pairs[i][1] == "update_status" and pairs[i + 1][1] == "add_subtask":
                pytest.fail(f"event order at step {pairs[i][0]} contradicts file order")
            break
    else:
        pytest.skip("no same-step events in this run")
