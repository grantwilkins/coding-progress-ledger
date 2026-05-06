from pathlib import Path

import csv
import json

import pytest

from agent_migrate_agent.adapters.swe_agent import swe_agent_to_trace
from agent_migrate_agent.cut_points import (
    CutPoint,
    PHASES,
    classify_phase,
    find_cut_points,
    load_trace_jsonl,
    write_cut_points_csv,
)

FIXTURE = Path(__file__).parent / "fixtures" / "swe_agent_pilot_s_07.json"
H5A_TRACE = Path(__file__).resolve().parents[1] / "examples" / "traces" / "h5a_multi_trajectory_swe.jsonl"


def _f2_events(tmp_path: Path) -> list[dict]:
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    return load_trace_jsonl(out)


# ---------------------------------------------------------------------------
# Predicate tests on the F2 fixture
# ---------------------------------------------------------------------------


def test_each_cut_point_has_llm_call_at_index(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    assert cps, "F2 fixture must produce at least one cut point"
    for cp in cps:
        e = events[cp.event_index]
        assert e["event_type"] == "add_subtask"
        assert e["payload"]["node_type"] == "llm_call"
        assert e["subtask_id"] == cp.next_llm_call_id


def test_no_open_subtask_at_cut(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    for cp in cps:
        open_set: set[str] = set()
        for e in events[: cp.event_index]:
            sid = e.get("subtask_id")
            if not sid:
                continue
            if e["event_type"] == "add_subtask":
                open_set.add(sid)
            elif e["event_type"] == "update_status" and (e.get("payload") or {}).get("status") == "complete":
                open_set.discard(sid)
        assert not open_set, f"cut at {cp.event_index} has open subtasks: {open_set}"


def test_every_declared_state_has_content_hash_at_cut(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    for cp in cps:
        for e in events[: cp.event_index]:
            if e["event_type"] == "state_declare":
                ch = (e.get("payload") or {}).get("content_hash")
                assert ch, f"state_declare missing content_hash before cut at {cp.event_index}"


def test_cut_points_cover_at_least_two_phases_on_f2(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    phases = {cp.phase for cp in cps}
    assert len(phases) >= 2
    assert phases.issubset(set(PHASES))


def test_one_cut_point_per_inter_llm_gap_per_session(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    total_llm_calls = sum(
        1 for e in events
        if e["event_type"] == "add_subtask" and e["payload"]["node_type"] == "llm_call"
    )
    assert len(cps) == total_llm_calls - 1


def test_cut_point_ids_chain_correctly(tmp_path: Path):
    """Pin the (prior, next) chain to catch reassignment_order regressions."""
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    for k, cp in enumerate(cps, start=2):
        assert cp.prior_llm_call_id == f"S{k - 1}"
        assert cp.next_llm_call_id == f"S{k}"
        assert cp.next_llm_call_ordinal == k


def test_cut_points_carry_workflow_and_session(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    assert all(cp.workflow_id.startswith("swe_agent_") for cp in cps)
    assert all(cp.session_id == "mc706__changelog-cli-34" for cp in cps)
    assert all(cp.prefix_tokens > 0 for cp in cps)
    assert all(cp.last_state_declared for cp in cps)


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------


def test_single_llm_call_yields_no_cut_points():
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 3, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
    ]
    assert find_cut_points(events, trace_id="t") == []


def test_empty_event_list_yields_no_cut_points():
    assert find_cut_points([], trace_id="t") == []


def test_init_only_yields_no_cut_points():
    events = [{"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None}]
    assert find_cut_points(events, trace_id="t") == []


def test_in_flight_tool_call_excludes_cut_point():
    events = [
        {"step": 0, "event_type": "init", "subtask_id": None, "payload": {}, "reason": None},
        {"step": 1, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 3, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "T1",
         "payload": {"node_type": "tool_call"}, "reason": None},
        {"step": 5, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 6, "event_type": "update_status", "subtask_id": "T1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 7, "event_type": "update_status", "subtask_id": "S2",
         "payload": {"status": "complete"}, "reason": None},
    ]
    cps = find_cut_points(events, trace_id="t")
    assert cps == [], "S2 must not be a cut point: T1 is in_flight at index 5"


def test_in_progress_status_does_not_close_subtask():
    """An update_status with status != 'complete' must NOT clear open_subtasks."""
    events = [
        {"step": 0, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 2, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "in_progress"}, "reason": None},
        {"step": 3, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
    ]
    cps = find_cut_points(events, trace_id="t")
    assert cps == [], "S1 still in_progress; S2 must not be a cut point"


def test_missing_content_hash_excludes_cut_point():
    events = [
        {"step": 0, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "", "tokens": 1}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 2, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 3, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "q", "content_hash": "hq", "tokens": 1}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
    ]
    assert find_cut_points(events, trace_id="t") == []


def test_state_invalidate_does_not_block_cut_point():
    """state_invalidate between calls must not block the cut: the prior content_hash is still recorded."""
    events = [
        {"step": 0, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h_p", "tokens": 1}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
        {"step": 2, "event_type": "update_status", "subtask_id": "S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 3, "event_type": "state_invalidate", "subtask_id": None,
         "payload": {"state_id": "p", "reason": "stale"}, "reason": None},
        {"step": 4, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"node_type": "llm_call", "session_id": "x"}, "reason": None},
    ]
    cps = find_cut_points(events, trace_id="t")
    assert len(cps) == 1
    assert cps[0].next_llm_call_id == "S2"


def test_cuts_are_scoped_per_session_no_inter_session_cut():
    """A trace with two sessions back_to_back must NOT emit a cut between sessions."""
    events = [
        {"step": 0, "event_type": "state_declare", "subtask_id": None,
         "payload": {"state_id": "p", "content_hash": "h", "tokens": 1}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "A_S1",
         "payload": {"node_type": "llm_call", "session_id": "A"}, "reason": None},
        {"step": 2, "event_type": "update_status", "subtask_id": "A_S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 3, "event_type": "add_subtask", "subtask_id": "B_S1",
         "payload": {"node_type": "llm_call", "session_id": "B"}, "reason": None},
        {"step": 4, "event_type": "update_status", "subtask_id": "B_S1",
         "payload": {"status": "complete"}, "reason": None},
        {"step": 5, "event_type": "add_subtask", "subtask_id": "B_S2",
         "payload": {"node_type": "llm_call", "session_id": "B"}, "reason": None},
    ]
    cps = find_cut_points(events, trace_id="t")
    assert len(cps) == 1
    assert cps[0].session_id == "B"
    assert cps[0].prior_llm_call_id == "B_S1"
    assert cps[0].next_llm_call_id == "B_S2"


def test_h5a_multi_session_cut_points_are_per_session():
    """h5a has 5 sessions × 2 llm_calls. Expected: 5 cut points, one per session."""
    if not H5A_TRACE.exists():
        pytest.skip(f"missing h5a fixture: {H5A_TRACE}")
    events = load_trace_jsonl(H5A_TRACE)
    cps = find_cut_points(events, trace_id="h5a")
    sessions = sorted({cp.session_id for cp in cps})
    assert sessions == ["cog", "dcj", "ice", "pok", "scf"]
    assert len(cps) == 5
    for cp in cps:
        assert cp.next_llm_call_ordinal == 2
        assert cp.total_llm_calls_in_session == 2
        assert cp.prior_llm_call_id == f"{cp.session_id}_S1"
        assert cp.next_llm_call_id == f"{cp.session_id}_S2"


def test_missing_event_type_hard_fails():
    events = [{"payload": {}}]
    with pytest.raises(ValueError, match="missing event_type"):
        find_cut_points(events, trace_id="t")


def test_load_trace_jsonl_includes_line_number_on_parse_error(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event_type":"init"}\n{not json}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_trace_jsonl(bad)


# ---------------------------------------------------------------------------
# CSV + phase classification
# ---------------------------------------------------------------------------


def test_csv_round_trip(tmp_path: Path):
    events = _f2_events(tmp_path)
    cps = find_cut_points(events, trace_id="s_07")
    out_csv = tmp_path / "out" / "s_07.csv"
    write_cut_points_csv(cps, out_csv)
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == len(cps)
    for cp, row in zip(cps, rows):
        assert int(row["event_index"]) == cp.event_index
        assert row["next_llm_call_id"] == cp.next_llm_call_id
        assert row["session_id"] == cp.session_id
        assert int(row["prefix_tokens"]) == cp.prefix_tokens
        assert row["phase"] == cp.phase


def test_phase_classification_thirds_l9():
    assert classify_phase(1, 9) == "early_exploration"
    assert classify_phase(3, 9) == "early_exploration"
    assert classify_phase(4, 9) == "mid_edit"
    assert classify_phase(6, 9) == "mid_edit"
    assert classify_phase(7, 9) == "pre_submit"
    assert classify_phase(9, 9) == "pre_submit"


def test_phase_classification_small_l():
    # L=2: only ordinal 2 is reachable, lands in pre_submit.
    assert classify_phase(2, 2) == "pre_submit"
    # L=3: third=1 → ordinal 1 early; 2 mid; 3 pre_submit.
    assert classify_phase(1, 3) == "early_exploration"
    assert classify_phase(2, 3) == "mid_edit"
    assert classify_phase(3, 3) == "pre_submit"
    # L=4: third=1 → ordinal 1 early, 2_3 mid, 4 pre_submit (asymmetric thirds at small L).
    assert classify_phase(1, 4) == "early_exploration"
    assert classify_phase(2, 4) == "mid_edit"
    assert classify_phase(3, 4) == "mid_edit"
    assert classify_phase(4, 4) == "pre_submit"
