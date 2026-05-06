"""C1 — cut_point detection over agent_migrate traces.

A cut point is a deterministic position in a recorded trajectory at which we
could pause and resume on a different site. Defined as `(trace_id, event_index)`
where `event_index` is the position of the `add_subtask` event for the next
`llm_call`, and the predicates:

  1. `events[event_index]` is `add_subtask` with `node_type == "llm_call"`.
  2. `event_index > 0` and a prior `llm_call` of the SAME `session_id` exists.
  3. No subtask is open at `event_index`. The prior `llm_call` has emitted
     `update_status` with `status == "complete"`, and no `tool_call` (or
     other) subtask is mid_flight. This is the in_flight check.
  4. Every `state_declare` event in `events[0:event_index]` carries a
     non_empty `content_hash`.

This is pure analysis over a JSONL trace. No model calls, no tool execution,
no real harness — § 0 forbids those. Cuts are scoped by `session_id` so that
multi_workflow traces (e.g. H5a) emit cuts within each session and not across
session boundaries.

Phase classification (`early_exploration | mid_edit | pre_submit`) is ordinal
thirds of the session's L total `llm_call`s; richer phase signals (e.g.
next_command = `submit`) require parsing the source trajectory and live in
C2/C3.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PHASES: tuple[str, ...] = ("early_exploration", "mid_edit", "pre_submit")


@dataclass(frozen=True)
class CutPoint:
    trace_id: str
    workflow_id: str
    session_id: str
    event_index: int
    prior_llm_call_id: str
    next_llm_call_id: str
    next_llm_call_ordinal: int
    total_llm_calls_in_session: int
    phase: str
    n_states_declared: int
    prefix_tokens: int
    last_state_declared: str


def find_cut_points(events: list[dict], trace_id: str) -> list[CutPoint]:
    """Return cut points found in `events`. Pure scan; no I/O.

    Hard_fails on events missing `event_type` (AGENTS.md: prefer hard fails).
    """
    for i, e in enumerate(events):
        if "event_type" not in e:
            raise ValueError(f"event {i} missing event_type: {e!r}")

    sessions_total: dict[str, int] = {}
    for e in events:
        if e["event_type"] == "add_subtask" and (e.get("payload") or {}).get("node_type") == "llm_call":
            sid = (e.get("payload") or {}).get("session_id") or ""
            sessions_total[sid] = sessions_total.get(sid, 0) + 1

    open_subtasks: set[str] = set()
    declared_hashes: dict[str, str] = {}
    declared_tokens: dict[str, int] = {}
    n_declared = 0
    last_state_declared = ""
    session_ordinal: dict[str, int] = {}
    session_prior_id: dict[str, str] = {}
    cut_points: list[CutPoint] = []

    for i, e in enumerate(events):
        et = e["event_type"]
        payload = e.get("payload") or {}
        sid = e.get("subtask_id")
        sid_str = str(sid) if sid is not None else ""

        if et == "add_subtask" and payload.get("node_type") == "llm_call":
            session = payload.get("session_id") or ""
            workflow = payload.get("workflow_id") or ""
            prior_in_session = session_prior_id.get(session)
            if (
                prior_in_session is not None
                and not open_subtasks
                and all(h for h in declared_hashes.values())
            ):
                ordinal = session_ordinal.get(session, 0) + 1
                total = sessions_total[session]
                cut_points.append(CutPoint(
                    trace_id=trace_id,
                    workflow_id=workflow,
                    session_id=session,
                    event_index=i,
                    prior_llm_call_id=prior_in_session,
                    next_llm_call_id=sid_str,
                    next_llm_call_ordinal=ordinal,
                    total_llm_calls_in_session=total,
                    phase=classify_phase(ordinal, total),
                    n_states_declared=n_declared,
                    prefix_tokens=sum(declared_tokens.values()),
                    last_state_declared=last_state_declared,
                ))
            session_ordinal[session] = session_ordinal.get(session, 0) + 1
            session_prior_id[session] = sid_str
            if sid_str:
                open_subtasks.add(sid_str)
        elif et == "add_subtask":
            if sid_str:
                open_subtasks.add(sid_str)
        elif et == "update_status":
            if payload.get("status") == "complete" and sid_str:
                open_subtasks.discard(sid_str)
        elif et == "state_declare":
            state_id = payload.get("state_id", "")
            declared_hashes[state_id] = payload.get("content_hash", "") or ""
            declared_tokens[state_id] = int(payload.get("tokens") or 0)
            n_declared += 1
            last_state_declared = state_id

    return cut_points


def classify_phase(ordinal: int, total: int) -> str:
    """Ordinal thirds: ordinal `k` of `total` llm_calls maps to a phase label."""
    third = max(1, total // 3)
    if ordinal <= third:
        return "early_exploration"
    if ordinal > total - third:
        return "pre_submit"
    return "mid_edit"


def write_cut_points_csv(cut_points: Iterable[CutPoint], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trace_id", "workflow_id", "session_id", "event_index",
            "prior_llm_call_id", "next_llm_call_id",
            "next_llm_call_ordinal", "total_llm_calls_in_session", "phase",
            "n_states_declared", "prefix_tokens", "last_state_declared",
        ])
        writer.writeheader()
        for cp in cut_points:
            writer.writerow(asdict(cp))


def load_trace_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    text = p.read_text()
    out: list[dict] = []
    for ln, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{ln}: malformed JSONL ({exc.msg})") from exc
    return out
