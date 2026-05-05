"""
Claim:
hermes_terminal_grader.grade_run emits a conservative verdict in
{failure, success_self_claim, ambiguous} per Hermes run, never asserts
success_verified, never sets proposed_final_success=True. Priority is
BLOCK -> failure(medium), budget -> failure(low), terminal ->
success_self_claim, else ambiguous.

Plausible wrong implementations:
- budget detector reads the wrong field on user-role events and misses budget exhaustion.
- terminal detector scans the whole trace, classifying mid-trajectory skill_view as terminal.
- grader sets proposed_final_success=True for self-claim (fabricates data per § 0.9).
- priority inverted: budget-exhausted run with terminal call graded failure not success_self_claim.
- evidence string shows frozenset() when last_action is None but terminal exists earlier in last 6.
- BLOCK fires on identical non-error tool responses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_estimator.labels.hermes_terminal_grader import (
    HermesGrade,
    grade_run,
    grade_source,
)


VALID_VERDICTS = {"failure", "success_self_claim", "ambiguous"}


def _write_run(tmp_path: Path, run_id: str, events: list[dict],
               issue: str = "test issue") -> Path:
    d = tmp_path / run_id
    d.mkdir()
    (d / "normalized_trace.json").write_text(json.dumps({
        "events": events,
        "issue_text": issue,
    }))
    return d


def _budget_user_event(text: str = "You've reached the maximum number of "
                                   "tool-calling iterations allowed.") -> dict:
    return {"role": "user", "observation": text}


def test_verdict_labels_are_disjoint_and_complete(tmp_path):
    fixtures = {
        "fail_budget": [{"role": "assistant", "action": "write_file"},
                        _budget_user_event()],
        "self_claim": [{"role": "assistant", "action": "submit_answer"}],
        "amb": [{"role": "assistant", "action": "terminal"}],
    }
    seen: set[str] = set()
    for rid, events in fixtures.items():
        d = _write_run(tmp_path, rid, events)
        g = grade_run(d)
        assert g.verdict in VALID_VERDICTS, g
        assert g.verdict != "success_verified"
        seen.add(g.verdict)
    assert seen == VALID_VERDICTS


def test_budget_exhausted_marks_failure_with_false(tmp_path):
    events = [
        {"role": "assistant", "action": "write_file"},
        {"role": "tool", "observation": "{\"bytes_written\": 100}"},
        _budget_user_event(),
    ]
    g = grade_run(_write_run(tmp_path, "r1", events))
    assert g.verdict == "failure"
    assert g.proposed_final_success is False
    assert any("budget_exhausted" in e for e in g.evidence)


def test_terminal_tool_call_yields_self_claim_with_null(tmp_path):
    for tool in ("skill_view", "submit_answer", "finish"):
        events = [
            {"role": "assistant", "action": "write_file"},
            {"role": "assistant", "action": tool},
        ]
        g = grade_run(_write_run(tmp_path, f"r_{tool}", events))
        assert g.verdict == "success_self_claim", (tool, g)
        # Critical: never True. Hermes has no verifier; True would fabricate.
        assert g.proposed_final_success is None, (tool, g)


def test_block_pattern_yields_failure_medium(tmp_path):
    err = "Error: missing dependency 'foo'"
    events = [
        {"role": "assistant", "action": "terminal"},
        {"role": "tool", "observation": err},
        {"role": "assistant", "action": "terminal"},
        {"role": "tool", "observation": err},
        {"role": "assistant", "action": "terminal"},
        {"role": "tool", "observation": err},
    ]
    g = grade_run(_write_run(tmp_path, "block", events))
    assert g.verdict == "failure"
    assert g.confidence == "medium"
    assert g.proposed_final_success is False


def test_block_does_not_fire_on_identical_non_error_acks(tmp_path):
    """3 identical successful 'ok' responses must not be graded failure."""
    ok = "{\"bytes_written\": 100, \"dirs_created\": false}"
    events = [
        {"role": "assistant", "action": "write_file"},
        {"role": "tool", "observation": ok},
        {"role": "assistant", "action": "write_file"},
        {"role": "tool", "observation": ok},
        {"role": "assistant", "action": "write_file"},
        {"role": "tool", "observation": ok},
    ]
    g = grade_run(_write_run(tmp_path, "okstreak", events))
    assert g.verdict == "ambiguous"


def test_ambiguous_default(tmp_path):
    events = [
        {"role": "assistant", "action": "search_files"},
        {"role": "tool", "observation": "{\"output\": \"...\"}"},
    ]
    g = grade_run(_write_run(tmp_path, "amb", events))
    assert g.verdict == "ambiguous"
    assert g.proposed_final_success is None


def test_terminal_priority_overrides_budget(tmp_path):
    """A run that hit budget but also issued a terminal call grades self_claim,
    not failure. Self-claim is the agent's last word."""
    events = [
        {"role": "assistant", "action": "skill_view"},
        _budget_user_event(),
    ]
    g = grade_run(_write_run(tmp_path, "tprio", events))
    assert g.verdict == "success_self_claim"
    assert g.proposed_final_success is None


def test_evidence_names_actual_terminal_tool_not_frozenset_literal(tmp_path):
    """Critic-flagged regression: when last_action is None but a terminal
    tool fired earlier in the last 6, the evidence string used to render
    `frozenset()` instead of the actual tool name."""
    events = [
        {"role": "assistant", "action": "skill_manage"},
        {"role": "tool", "observation": "ack"},
        {"role": "assistant", "action": None, "tool_name": None},  # thought-only
    ]
    g = grade_run(_write_run(tmp_path, "ev", events))
    assert g.verdict == "success_self_claim"
    joined = " ".join(g.evidence)
    assert "skill_manage" in joined
    assert "frozenset()" not in joined


def test_grade_source_skips_non_dirs_and_dirs_without_trace(tmp_path):
    _write_run(tmp_path, "real", [{"role": "assistant", "action": "submit_answer"}])
    (tmp_path / "stray.md").write_text("not a run")
    (tmp_path / "empty_dir").mkdir()
    out = grade_source(tmp_path)
    assert [g.run_id for g in out] == ["real"]


def test_hermes_label_loader_still_unresolvable_for_v2():
    """State-of-the-world: until upstream T2 lands, Hermes must produce 0
    labels and 30 unresolvable runs (all None final_success). When T2
    lands, this test will go red and force a v1 review."""
    from coding_estimator.labels.build import build_source_labels
    df, stats = build_source_labels("hermes_pilot_h5_v2")
    assert df.empty
    assert stats.n_runs_total == 30
    assert stats.n_runs_labeled == 0
    assert stats.n_runs_unresolvable == 30
    assert stats.n_runs_malformed == 0
