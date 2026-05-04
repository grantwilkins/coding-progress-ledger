"""D0 golden fixture invariants.

The fixture itself is the executable definition of feature semantics.
Before the prefix replay engine (D2) or any feature builder (D3) lands,
this test pins the fixture's structural properties so the fixture cannot
silently rot.

Claim:
    1. Every required event_type appears in the canonical ledger.
    2. Both canonical and mutated ledgers parse via upstream
       load_events_jsonl.
    3. The two fixtures are byte-identical for every event with step
       <= t_mid; they diverge strictly past t_mid.
    4. expected_checkpoints.json has one entry per step in the canonical
       ledger, contiguous from 0 to terminal_step.
    5. Hand-derived expected aggregates agree with upstream replay+score
       at every step (defense against typo-in-the-fixture bugs).

Plausible wrong implementations of the FIXTURE itself:
    - missing required event types
    - mutated fixture diverges before t_mid (silent leakage in tests)
    - expected JSON skips a step or has off-by-one drops
    - hand value typo (e.g. drops_count = 4 at step 10 when truth is 5)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ledger_progress.core import EventType, replay
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score
from ledger_progress.serialization import load_events_jsonl

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "golden_run"
CANONICAL = FIXTURE_DIR / "ledger.jsonl"
MUTATED = FIXTURE_DIR / "ledger_mutated_after_t_mid.jsonl"
EXPECTED = FIXTURE_DIR / "expected_checkpoints.json"

REQUIRED_EVENT_TYPES = {
    EventType.INIT,
    EventType.ADD_SUBTASK,
    EventType.UPDATE_STATUS,
    EventType.ADD_EVIDENCE,
    EventType.SPLIT_SUBTASK,
    EventType.REOPEN_SUBTASK,
    EventType.INVALIDATE_SUBTASK,
}

REQUIRED_STATUS_TRANSITIONS = {"in_progress", "complete", "blocked"}


def _expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def test_fixture_files_exist() -> None:
    assert CANONICAL.is_file()
    assert MUTATED.is_file()
    assert EXPECTED.is_file()


def test_canonical_parses_via_upstream() -> None:
    events = load_events_jsonl(str(CANONICAL))
    assert len(events) >= 14
    types_seen = {e.event_type for e in events}
    missing = REQUIRED_EVENT_TYPES - types_seen
    assert not missing, f"fixture missing required event types: {missing}"


def test_required_status_transitions_present() -> None:
    events = load_events_jsonl(str(CANONICAL))
    seen = {
        e.payload["status"]
        for e in events
        if e.event_type is EventType.UPDATE_STATUS and "status" in e.payload
    }
    missing = REQUIRED_STATUS_TRANSITIONS - seen
    assert not missing, f"fixture missing status transitions: {missing}"


def test_validation_pass_and_fail_present() -> None:
    """A validation leaf must be marked complete (pass) AND invalidated
    (fail) within the fixture; both signals are needed for downstream
    validation-feature tests."""
    events = load_events_jsonl(str(CANONICAL))
    val_subtask_ids = {
        e.subtask_id
        for e in events
        if e.event_type is EventType.ADD_SUBTASK
        and e.payload.get("category") == "validation"
    }
    pass_seen = any(
        e.event_type is EventType.UPDATE_STATUS
        and e.subtask_id in val_subtask_ids
        and e.payload.get("status") == "complete"
        for e in events
    )
    fail_seen = any(
        e.event_type is EventType.INVALIDATE_SUBTASK and e.subtask_id in val_subtask_ids
        for e in events
    )
    assert pass_seen, "fixture must include a validation pass"
    assert fail_seen, "fixture must include a validation fail"


def test_mutated_fixture_diverges_only_after_t_mid() -> None:
    canonical = CANONICAL.read_text(encoding="utf-8").splitlines()
    mutated = MUTATED.read_text(encoding="utf-8").splitlines()
    t_mid = _expected()["t_mid"]
    canonical_events = load_events_jsonl(str(CANONICAL))
    mutated_events = load_events_jsonl(str(MUTATED))

    # File-line equality up through every event with step <= t_mid.
    canonical_prefix = [
        line for line, e in zip(canonical, canonical_events, strict=True) if e.step <= t_mid
    ]
    mutated_prefix = [
        line for line, e in zip(mutated, mutated_events, strict=True) if e.step <= t_mid
    ]
    assert canonical_prefix == mutated_prefix, (
        "canonical and mutated fixtures MUST be byte-identical through t_mid; "
        "any divergence here is a leakage hazard for downstream tests"
    )

    # And they must actually differ past t_mid -- otherwise the mutated
    # fixture is doing nothing useful.
    canonical_tail = [
        line for line, e in zip(canonical, canonical_events, strict=True) if e.step > t_mid
    ]
    mutated_tail = [
        line for line, e in zip(mutated, mutated_events, strict=True) if e.step > t_mid
    ]
    assert canonical_tail != mutated_tail, "mutated fixture's tail must differ from canonical"


def test_expected_checkpoints_contiguous() -> None:
    exp = _expected()
    steps = [c["step"] for c in exp["checkpoints"]]
    assert steps == list(range(0, exp["terminal_step"] + 1))


@pytest.mark.parametrize(
    "fixture_path,fixture_name",
    [(CANONICAL, "canonical"), (MUTATED, "mutated")],
)
def test_fixture_terminal_step_matches_max_event(fixture_path: Path, fixture_name: str) -> None:
    events = load_events_jsonl(str(fixture_path))
    assert max(e.step for e in events) == _expected()["terminal_step"], fixture_name


def test_hand_derived_expected_agrees_with_upstream() -> None:
    """Cross-check every per-step aggregate against upstream replay+score.
    A typo in the hand-derived JSON is a real risk; this test catches it
    immediately, before any feature builder consumes the fixture."""
    events = load_events_jsonl(str(CANONICAL))
    exp_by_step = {c["step"]: c for c in _expected()["checkpoints"]}

    prev = 0.0
    drops = 0
    largest = 0.0
    adds = splits = reopens = invals = 0

    for t in range(0, max(e.step for e in events) + 1):
        for e in [ev for ev in events if ev.step == t]:
            if e.event_type is EventType.ADD_SUBTASK:
                adds += 1
            elif e.event_type is EventType.SPLIT_SUBTASK:
                splits += 1
            elif e.event_type is EventType.REOPEN_SUBTASK:
                reopens += 1
            elif e.event_type is EventType.INVALIDATE_SUBTASK:
                invals += 1
        prefix = [ev for ev in events if ev.step <= t]
        obs = score(replay(prefix), categories=CODING_CATEGORIES)
        drop = max(0.0, prev - obs.progress)
        if drop > 1e-9:
            drops += 1
            largest = max(largest, drop)
        prev = obs.progress

        exp = exp_by_step[t]
        assert exp["active_leaf_count"] == obs.active_leaf_count, t
        assert exp["completed_leaf_count"] == obs.complete_leaf_count, t
        assert abs(exp["coding_progress"] - obs.progress) < 1e-9, t
        assert exp["num_adds_so_far"] == adds, t
        assert exp["num_splits_so_far"] == splits, t
        assert exp["num_reopens_so_far"] == reopens, t
        assert exp["num_invalidations_so_far"] == invals, t
        assert exp["num_progress_drops_so_far"] == drops, t
        assert abs(exp["largest_progress_drop_so_far"] - largest) < 1e-9, t


def test_t_mid_falls_inside_run() -> None:
    exp = _expected()
    assert 0 < exp["t_mid"] < exp["terminal_step"], (
        "t_mid must be strictly inside the run for future-mutation tests "
        "to have anything meaningful to compare"
    )


def test_largest_drop_at_t_mid_is_pinned() -> None:
    """Pin a hand-checkable invariant: at t_mid=10, the largest progress
    drop seen so far is 0.5 (occurring at step 3 when s2 was added on top
    of a fully-complete s1). If a future fixture edit changes this,
    expected_checkpoints.json MUST be updated in lockstep."""
    exp = _expected()
    at_t_mid = next(c for c in exp["checkpoints"] if c["step"] == exp["t_mid"])
    assert at_t_mid["largest_progress_drop_so_far"] == 0.5
    assert at_t_mid["num_progress_drops_so_far"] == 5
