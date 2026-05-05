"""Conservative deterministic grader for Hermes `final_success` proposals.

Hermes traces ship no upstream eval logs; `final_success` is genuinely
subjective and requires human review. This grader emits one of:

    - "failure": evidence of an unrecoverable terminal state.
    - "success_self_claim": agent reached a terminal tool call and produced
      observable artifacts; the trajectory is consistent with success but
      not verified.
    - "ambiguous": neither failure evidence nor a terminal claim — must be
      reviewed by a human.

Never emits "success_verified". Hermes provides no verifier, so verified
success is unreachable from the trace alone.

The grader exists so the upstream annotator (T2) has a per-run starting
point with explicit evidence. It does **not** ship labels into upstream
`source_metadata.json` directly; the annotator decides whether to accept,
override, or escalate each proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TERMINAL_TOOLS = frozenset(
    {"submit_answer", "final_response", "task_complete", "finish",
     "skill_manage", "skill_view"}
)
BUDGET_EXHAUSTED_RE = "maximum number of tool-calling iterations"


@dataclass(frozen=True)
class HermesGrade:
    run_id: str
    verdict: str  # "failure" | "success_self_claim" | "ambiguous"
    proposed_final_success: bool | None
    confidence: str  # "low" | "medium"
    evidence: tuple[str, ...]
    issue: str
    n_steps: int
    last_assistant_action: str | None


def _last_assistant_action(events: list[dict]) -> str | None:
    for e in reversed(events):
        if e.get("role") == "assistant":
            return e.get("action") or e.get("tool_name")
    return None


def _budget_exhausted(events: list[dict]) -> bool:
    for e in events[-4:]:
        blob = " ".join(
            str(e.get(k, "")) for k in ("content", "text", "observation", "command")
        )
        if BUDGET_EXHAUSTED_RE in blob:
            return True
    return False


def _has_terminal_call(events: list[dict]) -> bool:
    for e in events[-6:]:
        if e.get("role") != "assistant":
            continue
        action = e.get("action") or e.get("tool_name")
        if action in TERMINAL_TOOLS:
            return True
    return False


def _repeated_terminal_errors(events: list[dict]) -> bool:
    tail = [
        str(e.get("observation", "")) or str(e.get("content", ""))
        for e in events[-8:] if e.get("role") == "tool"
    ]
    if len(tail) < 3:
        return False
    last3 = tail[-3:]
    if len(set(last3)) > 1:
        return False
    return any(s in last3[-1].lower() for s in ("error", "failed", "missing"))


def grade_run(run_dir: Path) -> HermesGrade:
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    nt = json.loads((run_dir / "normalized_trace.json").read_text())
    events = nt.get("events", [])
    issue = nt.get("issue_text", "")
    n = len(events)
    last_action = _last_assistant_action(events)
    budget = _budget_exhausted(events)
    terminal = _has_terminal_call(events)
    repeated_err = _repeated_terminal_errors(events)

    evidence: list[str] = []
    if budget:
        evidence.append("budget_exhausted: trajectory hit the iteration limit")
    if terminal:
        terminal_names = sorted(
            {(e.get("action") or e.get("tool_name"))
             for e in events[-6:]
             if e.get("role") == "assistant"
             and (e.get("action") or e.get("tool_name")) in TERMINAL_TOOLS}
        )
        evidence.append(
            f"terminal_tool_call: last 6 steps include {terminal_names}"
        )
    if repeated_err:
        evidence.append("repeated_terminal_errors: last 3 tool responses identical and error-shaped")
    if last_action is None:
        evidence.append("trajectory ends in (thought-only) — no final tool call")

    if repeated_err and not terminal:
        return HermesGrade(
            run_id=run_dir.name,
            verdict="failure",
            proposed_final_success=False,
            confidence="medium",
            evidence=tuple(evidence),
            issue=issue,
            n_steps=n,
            last_assistant_action=last_action,
        )
    if budget and not terminal:
        return HermesGrade(
            run_id=run_dir.name,
            verdict="failure",
            proposed_final_success=False,
            confidence="low",
            evidence=tuple(evidence),
            issue=issue,
            n_steps=n,
            last_assistant_action=last_action,
        )
    if terminal:
        return HermesGrade(
            run_id=run_dir.name,
            verdict="success_self_claim",
            proposed_final_success=None,
            confidence="low",
            evidence=tuple(evidence),
            issue=issue,
            n_steps=n,
            last_assistant_action=last_action,
        )
    return HermesGrade(
        run_id=run_dir.name,
        verdict="ambiguous",
        proposed_final_success=None,
        confidence="low",
        evidence=tuple(evidence),
        issue=issue,
        n_steps=n,
        last_assistant_action=last_action,
    )


def grade_source(source_runs_dir: Path) -> list[HermesGrade]:
    if not source_runs_dir.is_dir():
        raise NotADirectoryError(source_runs_dir)
    out: list[HermesGrade] = []
    for d in sorted(source_runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "normalized_trace.json").is_file():
            continue
        out.append(grade_run(d))
    return out
