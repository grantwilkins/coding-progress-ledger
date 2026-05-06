"""SWE_agent trajectory adapter.

Converts a single SWE_agent JSON trajectory (one linear LLM session) into a
agent_migrate trace: one `add_subtask` per `ai` turn (an llm_call), plus
`state_declare`/`state_read` events for the three shared_state classes that
exist in a SWE_agent session:

  1. `system_prompt` — the same SETTING text on every turn, consumed by every
     llm_call.
  2. `issue_text` — the first `user` turn (issue + instructions), also
     consumed by every llm_call.
  3. `tool_output_<hash>` — `user` turn text following each `ai` turn (i.e.,
     the tool output produced by running the previous command).

**Accumulation model.** A tool output produced at turn N is read by every
ai turn N+1, N+2, ..., end_of_trajectory. This matches LLM context_window
behavior: each ai response sees the full prior conversation. Without this,
F2 understates per_state consumer counts dramatically and only counts
sharing across byte_identical content_hash collisions.

**Content_hash dedup.** Two distinct tool outputs that happen to have the
same content (e.g., two `cat empty.py` calls) merge into one state object.
This is correct for agent_migrate's "could a cache serve it" semantic; it is wrong
for any analysis treating state_id as temporal identity ("the same artifact
mutating over time").

Token counts are approximated as `len(text) // 4` (CLAUDE.md disallows
adding tiktoken). On code_heavy text the real ratio is ~3.3_3.6, so this
under_counts by ~15_20%. Approximations are fine for ranking modes/sites
WITHIN one trace; do NOT use them for cross_trace numeric comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import events as v_events
from ..hashing import segment_hash


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def swe_agent_to_events(traj_path: str | Path, workflow_id: str | None = None) -> list[dict]:
    raw = json.loads(Path(traj_path).read_text())
    instance_id = raw["instance_id"]
    workflow_id = workflow_id or f"swe_agent_{instance_id}"
    turns = raw["trajectory"]

    if not turns or turns[0].get("role") != "system":
        raise ValueError(f"first turn must be system in {traj_path}")

    system_prompt = turns[0].get("system_prompt") or ""
    if not system_prompt:
        raise ValueError(f"system turn has no system_prompt in {traj_path}")

    if turns[1].get("role") != "user":
        raise ValueError(f"first non_system turn must be user (issue text) in {traj_path}")

    issue_text = _first_user_text(turns)
    if not issue_text:
        raise ValueError(f"no first user turn (issue text) in {traj_path}")

    ev: list[dict] = []
    step = 0

    def emit(event_type: str, subtask_id: str | None, payload: dict, reason: str | None = None) -> None:
        nonlocal step
        ev.append({"step": step, "event_type": event_type, "subtask_id": subtask_id,
                   "payload": payload, "reason": reason})
        step += 1

    emit("init", None, {"root_task": f"swe_agent: {instance_id}"})

    sys_state = {
        "state_id": "system_prompt",
        "content_hash": segment_hash(system_prompt),
        "layer": "prompt_context",
        "lifetime": "persistent",
        "tokens": approx_tokens(system_prompt),
        "bytes": None,
        "producer_node_id": None,
    }
    emit(v_events.STATE_DECLARE, None, sys_state)

    issue_state = {
        "state_id": "issue_text",
        "content_hash": segment_hash(issue_text),
        "layer": "prompt_context",
        "lifetime": "shared",
        "tokens": approx_tokens(issue_text),
        "bytes": None,
        "producer_node_id": None,
    }
    emit(v_events.STATE_DECLARE, None, issue_state)

    declared_outputs: dict[str, str] = {}        # content_hash -> state_id
    state_meta: dict[str, dict] = {}              # state_id -> {"hash": ..., "tokens": ...}
    state_meta["system_prompt"] = {"hash": sys_state["content_hash"], "tokens": sys_state["tokens"]}
    state_meta["issue_text"] = {"hash": issue_state["content_hash"], "tokens": issue_state["tokens"]}
    issue_hash = segment_hash(issue_text)
    declared_outputs[issue_hash] = "issue_text"

    llm_call_index = 0
    accumulated_output_state_ids: list[str] = []

    for i, turn in enumerate(turns[1:]):
        role = turn.get("role")
        text = turn.get("text") or ""

        if role == "user":
            if i == 0:
                continue
            if not text:
                continue
            content_hash = segment_hash(text)
            if content_hash in declared_outputs:
                state_id = declared_outputs[content_hash]
            else:
                state_id = f"tool_output_{len(declared_outputs) - 1:03d}"
                declared_outputs[content_hash] = state_id
                state_tokens = approx_tokens(text)
                state_meta[state_id] = {"hash": content_hash, "tokens": state_tokens}
                emit(v_events.STATE_DECLARE, None, {
                    "state_id": state_id,
                    "content_hash": content_hash,
                    "layer": "prompt_context",
                    "lifetime": "ephemeral",
                    "tokens": state_tokens,
                    "bytes": None,
                    "producer_node_id": None,
                })
            if state_id not in accumulated_output_state_ids:
                accumulated_output_state_ids.append(state_id)
            continue

        if role == "ai":
            llm_call_index += 1
            node_id = f"S{llm_call_index}"
            emit("add_subtask", node_id, {
                "description": f"llm_call_{llm_call_index}",
                "parent_id": None,
                "weight": 1.0,
                "category": "product",
                "node_type": "llm_call",
                "workflow_id": workflow_id,
                "session_id": instance_id,
            })
            for sid in ("system_prompt", "issue_text", *accumulated_output_state_ids):
                meta = state_meta[sid]
                emit(v_events.STATE_READ, None, {
                    "state_id": sid,
                    "content_hash": meta["hash"],
                    "consumer_node_id": node_id,
                    "tokens": meta["tokens"],
                })
            emit("update_status", node_id, {
                "status": "complete",
                "evidence": [f"{node_id} response emitted"],
            })
            continue

    return ev


def swe_agent_to_trace(traj_path: str | Path, out_path: str | Path,
                       workflow_id: str | None = None) -> list[dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events = swe_agent_to_events(traj_path, workflow_id=workflow_id)
    out_path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events))
    return events


def _first_user_text(turns: list[dict]) -> str:
    for turn in turns:
        if turn.get("role") == "user":
            return turn.get("text") or ""
    return ""
