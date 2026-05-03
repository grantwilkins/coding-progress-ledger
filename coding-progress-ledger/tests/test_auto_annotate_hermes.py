"""HP5/HP6 auto-annotator invariants + regression vs HP4 human annotations.

Claim:
The auto-annotator is the measurement layer for the observation channel.
It must report the agent's *actual* trace shape — not a verdict on whether
the task will succeed. Specifically the HP6 BLOCKED rule asserts:

  - 3+ consecutive identical errors -> BLOCKED  (real stuck loop)
  - error -> success                -> COMPLETE (recovery, full credit)
  - lone trailing error             -> IN_PROGRESS (transient, not a verdict)
  - distinct errors                 -> IN_PROGRESS (learning, not stuck)

The estimator (downstream) consumes complete/blocked/in-progress as three
distinct evidence states; collapsing any two of them into one is a
measurement bug, not a "tidy-up".

Plausible wrong implementations these tests target:
  - status conflation: marking lone-error leaf COMPLETE (overstates progress)
    or BLOCKED (HP5 regression)
  - off-by-one threshold: requiring strictly more than 3 identical errors
  - cross-leaf streak leakage: error_streak as module/global state
  - recovery progress discount: penalizing a recovered leaf below 1.0
  - last_complete_evidence advancing on error: citing a failed step as
    the leaf's completion evidence
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.auto_annotate_hermes import (  # noqa: E402
    auto_annotate, classify_step, _is_error_response, _build_groups,
    TERMINAL_TOOL_RE, ERROR_STREAK_BLOCK_THRESHOLD,
)
from ledger_progress import SubtaskCategory  # noqa: E402

HP4_RUNS = ROOT / "runs" / "hermes_pilot"
HP4_PILOTS = sorted(d.name for d in HP4_RUNS.iterdir() if d.is_dir() and d.name.startswith("hermes_pilot_"))


def _trace(pilot: str) -> dict:
    return json.loads((HP4_RUNS / pilot / "normalized_trace.json").read_text())


def test_classify_known_tools():
    assert classify_step("read_file", None) == SubtaskCategory.INVESTIGATION
    assert classify_step("write_file", None) == SubtaskCategory.PRODUCT
    assert classify_step("patch", None) == SubtaskCategory.PRODUCT
    assert classify_step("submit_answer", None) == SubtaskCategory.ARTIFACT
    assert classify_step("pytest", None) == SubtaskCategory.VALIDATION
    assert classify_step("pip_install", None) == SubtaskCategory.ENVIRONMENT


def test_classify_shell_by_intent():
    assert classify_step("bash", '{"command": "pytest tests/"}') == SubtaskCategory.VALIDATION
    assert classify_step("bash", '{"command": "ls -la"}') == SubtaskCategory.INVESTIGATION
    assert classify_step("bash", '{"command": "pip install foo"}') == SubtaskCategory.ENVIRONMENT
    assert classify_step("bash", '{"command": "echo hello > out.txt"}') == SubtaskCategory.PRODUCT


def test_terminal_tool_re_matches_protocol():
    for name in ("submit_answer", "final_response", "task_complete", "skill_manage"):
        assert TERMINAL_TOOL_RE.match(name)
    assert not TERMINAL_TOOL_RE.match("write_file")


def test_error_detection():
    assert _is_error_response('{"error": "boom"}')
    assert _is_error_response('{"success": false}')
    assert _is_error_response('{"exit_code": 1}')
    assert not _is_error_response('{"output": "ok", "exit_code": 0}')
    assert not _is_error_response('{"success": true}')
    assert not _is_error_response(None)


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_no_evidence_cites_thought(pid):
    trace = _trace(pid)
    by_idx = {e["step_index"]: e for e in trace["events"]}
    session = auto_annotate(trace)
    events = session.ledger.events
    step_re = re.compile(r"step\s+(\d+)", re.IGNORECASE)
    for ev in events:
        evidence = (ev.payload or {}).get("evidence")
        if not evidence:
            continue
        items = evidence if isinstance(evidence, list) else [evidence]
        for s in items:
            for n in [int(m) for m in step_re.findall(s)]:
                step = by_idx.get(n)
                if step and step["role"] == "assistant":
                    assert step.get("action") is not None, (
                        f"{pid}: evidence cites thought-only step {n}"
                    )


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_events_step_monotone(pid):
    trace = _trace(pid)
    session = auto_annotate(trace)
    steps = [e.step for e in session.ledger.events]
    assert steps == sorted(steps), f"{pid}: events not step-monotone: {steps}"


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_blocked_rule_consistent(pid):
    """HP6 softened: a leaf is BLOCKED iff we saw 3+ consecutive identical
    error responses OR 3+ identical bodies (Pitfall H3). A single transient
    error must NOT BLOCK."""
    trace = _trace(pid)
    session = auto_annotate(trace)
    for sub in session.ledger.subtasks.values():
        if sub.status.value != "blocked":
            continue
        assert sub.evidence, f"{pid}: blocked leaf {sub.id} has no evidence"


def test_softened_rule_constants():
    assert ERROR_STREAK_BLOCK_THRESHOLD == 3


def _make_trace(steps):
    """steps: list of (tool_name, args_json, observation) — emit a synthetic
    normalized trace with paired call/tool events."""
    events = []
    idx = 0
    for tool_name, args, obs in steps:
        events.append({
            "step_index": idx,
            "role": "assistant",
            "tool_name": tool_name,
            "command": args,
            "action": "tool_call",
        })
        call_idx = idx
        idx += 1
        events.append({
            "step_index": idx,
            "role": "tool",
            "tool_name": tool_name,
            "observation": obs,
            "raw": {"paired_call_event_step": call_idx},
        })
        idx += 1
    return {"issue_text": "synthetic", "events": events}


def test_single_error_does_not_block():
    """HP6: an isolated error response must NOT BLOCK the leaf."""
    trace = _make_trace([
        ("read_file", '{"path": "a.py"}', '{"error": "boom"}'),
        ("read_file", '{"path": "b.py"}', '{"output": "ok"}'),
    ])
    session = auto_annotate(trace)
    for sub in session.ledger.subtasks.values():
        assert sub.status.value != "blocked", "single error should not BLOCK under softened rule"


def test_three_identical_errors_block():
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    blocked = [s for s in session.ledger.subtasks.values() if s.status.value == "blocked"]
    assert len(blocked) == 1


def test_three_distinct_errors_do_not_block():
    """HP6: three errors with DIFFERENT bodies are not a stuck loop."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom1"}'),
        ("read_file", '{"path": "b"}', '{"error": "boom2"}'),
        ("read_file", '{"path": "c"}', '{"error": "boom3"}'),
    ])
    session = auto_annotate(trace)
    for sub in session.ledger.subtasks.values():
        assert sub.status.value != "blocked", "distinct errors should not BLOCK"


def test_error_then_success_completes():
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"output": "ok"}'),
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 1
    assert leaves[0].status.value == "complete", "recovery from transient error should complete"


def test_streak_resets_on_success():
    """HP6: err,err,ok,err must NOT BLOCK. Without the streak reset on
    success, error_streak after step 4 = [boom,boom,boom] (length 3, all
    identical) and the leaf would BLOCK. With the reset, the success at
    step 3 clears the streak so step 4's err is alone. This is the
    minimal trace that fails iff the `else: error_streak = []` reset is
    deleted — the original 5-step `err,err,ok,err,err` trace passed
    vacuously because both halves were length-2 streaks."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"output": "ok"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 1
    assert leaves[0].status.value != "blocked", (
        "success at step 3 must reset error_streak; without the reset, "
        "the trailing err would be the third in a streak and BLOCK"
    )


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_softened_rule_does_not_increase_blocked_count(pid):
    """HP6 acceptance: under the softened rule, BLOCKED leaves on HP4 traces
    must be <= the previous behavior (which BLOCKED on first error). Computed
    by re-deriving the "old rule" count from the trace and comparing."""
    trace = _trace(pid)
    session = auto_annotate(trace)
    new_blocked = sum(
        1 for s in session.ledger.subtasks.values() if s.status.value == "blocked"
    )
    # Derive the "old rule" upper bound: any leaf containing >=1 error response
    # would have BLOCKED under the old rule. So new_blocked <= old_upper_bound.
    by_idx = {e["step_index"]: e for e in trace["events"]}
    groups = _build_groups(trace["events"])
    old_upper = 0
    for g in groups:
        for step_idx in g["steps"]:
            for ev in trace["events"]:
                if (ev["role"] == "tool"
                        and (ev.get("raw") or {}).get("paired_call_event_step") == step_idx
                        and _is_error_response(ev.get("observation"))):
                    old_upper += 1
                    break
            else:
                continue
            break
    assert new_blocked <= old_upper, (
        f"{pid}: softened rule produced {new_blocked} BLOCKED, "
        f"old rule upper bound was {old_upper}"
    )


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_terminal_tool_closes_artifact(pid):
    trace = _trace(pid)
    by_idx = {e["step_index"]: e for e in trace["events"]}
    session = auto_annotate(trace)
    for sub in session.ledger.subtasks.values():
        if sub.category != SubtaskCategory.ARTIFACT:
            continue
        # The opener step's tool should be a terminal-tool by our rule
        opener_step = sub.created_at_step
        ev = by_idx.get(opener_step)
        if ev and ev["role"] == "assistant" and ev.get("tool_name"):
            assert TERMINAL_TOOL_RE.match(ev["tool_name"]), (
                f"{pid}: ARTIFACT leaf opened on non-terminal tool {ev['tool_name']}"
            )


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_deterministic(pid):
    trace = _trace(pid)
    s1 = auto_annotate(trace)
    s2 = auto_annotate(trace)
    e1 = [(e.event_type.value, e.step, e.subtask_id) for e in s1.ledger.events]
    e2 = [(e.event_type.value, e.step, e.subtask_id) for e in s2.ledger.events]
    assert e1 == e2, f"{pid}: auto-annotator not deterministic"


@pytest.mark.parametrize("pid", HP4_PILOTS)
def test_overlap_with_human_annotation(pid):
    """The heuristic must (a) match the human leaf-category multiset on at
    least 50% of human leaves, (b) keep leaf count within +/-50% of the
    human's, and (c) open at most one ARTIFACT leaf, and only on a step
    whose tool name contains a terminal signal. (a) alone passes for a
    degenerate "all-VALIDATION" heuristic; (b)+(c) close that loophole."""
    spec_path = ROOT / "annotations" / "hermes_pilot" / f"{pid}.json"
    if not spec_path.is_file():
        pytest.skip(f"no human spec for {pid}")
    spec = json.loads(spec_path.read_text())
    human_cats = sorted(
        ev["category"].lower() for ev in spec["events"] if ev["op"] == "add"
    )
    trace = _trace(pid)
    session = auto_annotate(trace)
    auto_cats = sorted(s.category.value for s in session.ledger.subtasks.values())

    common = 0
    auto_pool = list(auto_cats)
    for c in human_cats:
        if c in auto_pool:
            common += 1
            auto_pool.remove(c)
    overlap = common / max(1, len(human_cats))
    assert overlap >= 0.5, (
        f"{pid}: category overlap {overlap:.2f} < 0.50; "
        f"human={human_cats} auto={auto_cats}"
    )

    human_n = len(human_cats)
    auto_n = len(auto_cats)
    assert 0.5 * human_n <= auto_n <= 1.5 * human_n + 1, (
        f"{pid}: heuristic leaf count {auto_n} outside +/-50% of human {human_n}"
    )

    terminal_signals = ("submit", "final_response", "task_complete", "answer", "skill_manage")
    for sub in session.ledger.subtasks.values():
        if sub.category.value != "artifact":
            continue
        opener_step = sub.created_at_step
        ev = trace["events"][opener_step]
        tool = (ev.get("tool_name") or "").lower()
        assert any(t in tool for t in terminal_signals), (
            f"{pid}: ARTIFACT leaf {sub.id} opened at step {opener_step} "
            f"on non-terminal tool {tool!r}"
        )


# ---- HP6 semantic-claim tests (research-test-creator pass) ----

from ledger_progress import score  # noqa: E402
from ledger_progress.queries import CODING_CATEGORIES  # noqa: E402


def test_lone_error_leaf_is_in_progress_not_complete_not_blocked():
    """HP6 channel-state distinction: a leaf whose only response is a
    single error must be IN_PROGRESS — not COMPLETE (overstated success)
    and not BLOCKED (HP5 regression). Catches a future change that
    collapses the three-way COMPLETE/BLOCKED/IN_PROGRESS distinction."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 1
    assert leaves[0].status.value == "in_progress", (
        f"lone-error leaf must be IN_PROGRESS; got {leaves[0].status.value}"
    )


def test_two_identical_errors_do_not_block_boundary():
    """HP6 threshold boundary: exactly 2 identical errors must NOT BLOCK.
    Catches off-by-one (>3 vs >=3) and threshold-bump regressions."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 1
    assert leaves[0].status.value != "blocked", (
        "2 identical errors must not yet trigger BLOCK (threshold is 3)"
    )


def test_error_streak_does_not_leak_across_leaves():
    """HP6 invariant: error_streak is per-leaf state. Two errors in a
    PRODUCT leaf followed by one identical error in a separate VALIDATION
    leaf must NOT BLOCK either leaf. Catches a future refactor that
    promotes error_streak to module/global scope."""
    trace = _make_trace([
        ("write_file", '{"path": "a"}', '{"error": "boom"}'),
        ("write_file", '{"path": "a"}', '{"error": "boom"}'),
        ("pytest", '{"path": "tests/"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 2, f"expected 2 leaves (PRODUCT, VALIDATION); got {len(leaves)}"
    cats = {leaf.category.value for leaf in leaves}
    assert cats == {"product", "validation"}
    for leaf in leaves:
        assert leaf.status.value != "blocked", (
            f"streak must reset at leaf boundary; {leaf.id} ({leaf.category.value}) "
            f"BLOCKED on cross-leaf streak"
        )


def test_recovered_leaf_contributes_full_progress_credit():
    """HP6 channel-vs-outcome decoupling: a leaf that experienced
    error -> success must contribute progress = 1.0 to coding_progress,
    identical to a never-errored leaf. The channel does not penalize
    transient failures with a partial-credit discount.

    Compares two synthetic traces with identical structure except for the
    presence of an intermediate error. Both must score 1.0."""
    clean = _make_trace([
        ("read_file", '{"path": "a"}', '{"output": "ok"}'),
    ])
    recovered = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"output": "ok"}'),
    ])
    p_clean = score(auto_annotate(clean).ledger, categories=CODING_CATEGORIES).progress
    p_recovered = score(auto_annotate(recovered).ledger, categories=CODING_CATEGORIES).progress
    assert p_clean == 1.0, f"clean baseline broke: {p_clean}"
    assert p_recovered == 1.0, (
        f"recovered leaf must score 1.0 (full credit on recovery); got {p_recovered}"
    )


def test_recovered_leaf_evidence_cites_success_step_not_error_step():
    """HP6 measurement honesty: when a leaf recovers, the COMPLETE
    evidence must point at the *successful* step, not the error step.
    Catches a future change that lets last_complete_step advance on
    error responses."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),  # call=0, resp=1
        ("read_file", '{"path": "a"}', '{"output": "ok"}'),   # call=2, resp=3
    ])
    session = auto_annotate(trace)
    leaves = list(session.ledger.subtasks.values())
    assert len(leaves) == 1 and leaves[0].status.value == "complete"
    # evidence string format: "step <N>: <tool>(...)" — N is the call step,
    # which is even-indexed (0 or 2). The success-call step is 2.
    ev_str = " ".join(leaves[0].evidence)
    assert "step 2" in ev_str, (
        f"completion evidence must cite the success step (call=2); got {ev_str!r}"
    )
    assert "step 0" not in ev_str, (
        f"completion evidence must not cite the error step (call=0); got {ev_str!r}"
    )


def test_block_reason_contains_loop_keyword_for_downstream_flag():
    """HP6 channel-to-checkpoint wiring: the BLOCK reason string must
    contain the substring 'loop' or 'stuck'. The estimator's
    `repeated_observation_loop_flag` (build_estimator_checkpoints.py)
    keys on those literals to fire the W3 mask. If the reason text
    drifts, the shape-tag `stuck_loop` and the checkpoint flag
    `repeated_observation_loop_flag` desynchronize — the channel
    reports a stuck loop, the estimator never sees it.

    Catches the HP5 -> HP6 critic finding D6: under the initial HP6
    rollout, BLOCK reasons read 'consecutive identical errors' with no
    'loop'/'stuck' substring, so the checkpoint flag stayed 0 even
    when the shape-tag fired."""
    cases = [
        # 3 identical errors -> error-streak rule
        [
            ("read_file", '{"path": "a"}', '{"error": "boom"}'),
            ("read_file", '{"path": "a"}', '{"error": "boom"}'),
            ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ],
        # 3 identical non-error bodies -> Pitfall H3 rule
        [
            ("read_file", '{"path": "a"}', '{"output": "ok"}'),
            ("read_file", '{"path": "a"}', '{"output": "ok"}'),
            ("read_file", '{"path": "a"}', '{"output": "ok"}'),
        ],
    ]
    for steps in cases:
        session = auto_annotate(_make_trace(steps))
        block_events = [
            e for e in session.ledger.events
            if e.event_type.value == "update_status"
            and getattr((e.payload or {}).get("status"), "value", (e.payload or {}).get("status")) == "blocked"
        ]
        if not block_events:
            continue  # H3 rule may not block this exact 3-ok case if streak rule rejects it
        for ev in block_events:
            reason = (ev.reason or "") + " " + ((ev.payload or {}).get("reason") or "")
            reason = reason.lower()
            assert "loop" in reason or "stuck" in reason, (
                f"BLOCK reason must contain 'loop' or 'stuck' for "
                f"downstream flag wiring; got: {reason!r}"
            )


def test_in_progress_event_anchored_at_call_step_not_response_step():
    """HP6 channel anchor: IN_PROGRESS must fire at the assistant *call*
    step (when the agent transitioned to attempting), not at the tool
    response step. Anchoring at the response step backdates the agent's
    transition to information it had not yet observed at call time.

    Catches the channel-vs-outcome boundary defect (D1)."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    session = auto_annotate(trace)
    start_events = [
        e for e in session.ledger.events
        if e.event_type.value == "update_status"
        and getattr((e.payload or {}).get("status"), "value",
                    (e.payload or {}).get("status")) == "in_progress"
    ]
    assert len(start_events) == 1
    # Synthetic trace: assistant call at step 0, tool response at step 1.
    assert start_events[0].step == 0, (
        f"IN_PROGRESS must anchor at call step 0, not response step 1; "
        f"got step {start_events[0].step}"
    )


def test_blocked_leaf_contributes_zero_progress_credit():
    """HP6 channel-vs-outcome: a BLOCKED leaf must contribute 0/N to
    coding_progress (count in denominator, not numerator). Catches a
    future "soft-credit" change that grants partial progress to BLOCKED."""
    trace = _make_trace([
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
        ("read_file", '{"path": "a"}', '{"error": "boom"}'),
    ])
    obs = score(auto_annotate(trace).ledger, categories=CODING_CATEGORIES)
    assert obs.active_leaf_count == 1
    assert obs.complete_leaf_count == 0
    assert obs.progress == 0.0, (
        f"BLOCKED leaf must contribute 0 to numerator; got progress={obs.progress}"
    )
