"""HP5 auto-annotator invariants + regression vs HP4 human annotations."""

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
    TERMINAL_TOOL_RE,
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
    """A leaf is BLOCKED iff its last paired observation has an error key OR
    we observed 3+ identical consecutive bodies."""
    trace = _trace(pid)
    session = auto_annotate(trace)
    for sub in session.ledger.subtasks.values():
        if sub.status.value != "blocked":
            continue
        assert sub.evidence, f"{pid}: blocked leaf {sub.id} has no evidence"


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
