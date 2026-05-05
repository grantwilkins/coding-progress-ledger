"""Deterministic synthetic trace generator for the canonical toy scenario.

Scenario:

    parent planner reads shared system_prefix + shared repo_context.
    planner spawns N subagents (default 3: A, B, C).
    all subagents read shared system_prefix + shared repo_context.
    each subagent has its own private context.
    A and C share workspace; A reads, C reads + writes.
    parent merges results.

The generator is the source of truth. The committed JSONL is regeneratable
byte-for-byte from a fixed config + seed.

Wire format matches ledger_progress.serialization (step, event_type, subtask_id,
payload, reason, timestamp). Vagrant-specific event types ride alongside ledger
event types; replay requires the upstream pass-through hook (Workstream A2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import events as v_events


@dataclass(frozen=True)
class SyntheticConfig:
    num_subagents: int = 3
    shared_context_tokens: int = 8000
    private_context_tokens: tuple[int, ...] = (1500, 12000, 2000)
    workspace_bytes: int = 4_000_000
    system_prefix_tokens: int = 200
    seed: int = 0
    workflow_id: str = "toy_workflow_v1"
    root_task: str = "toy coding task"

    def __post_init__(self) -> None:
        if self.num_subagents != len(self.private_context_tokens):
            raise ValueError("private_context_tokens length must equal num_subagents")
        if self.num_subagents < 1:
            raise ValueError("num_subagents must be >= 1")


def _subagent_label(i: int) -> str:
    if i >= 26:
        raise ValueError("synthetic generator supports at most 26 subagents")
    return chr(ord("A") + i)


def _state(state_id: str, layer: str, lifetime: str, tokens: int = 0, bytes_: int | None = None,
           producer_node_id: str | None = None) -> dict:
    return {
        "state_id": state_id,
        "content_hash": f"hash_{state_id}_v1",
        "layer": layer,
        "lifetime": lifetime,
        "tokens": tokens,
        "bytes": bytes_,
        "producer_node_id": producer_node_id,
    }


def generate_events(config: SyntheticConfig) -> list[dict]:
    """Build the ordered event list for the canonical scenario."""
    cfg = config
    ev: list[dict] = []
    step = 0

    def emit(event_type: str, subtask_id: str | None, payload: dict, reason: str | None = None) -> None:
        nonlocal step
        ev.append({
            "step": step,
            "event_type": event_type,
            "subtask_id": subtask_id,
            "payload": payload,
            "reason": reason,
        })

    emit("init", None, {"root_task": cfg.root_task})
    step += 1

    planner = "S1"
    emit("add_subtask", planner, {
        "description": "planner",
        "parent_id": None,
        "weight": 1.0,
        "category": "product",
        "node_type": "llm_call",
        "workflow_id": cfg.workflow_id,
    })

    system_prefix = _state("system_prefix", "prompt_context", "shared",
                            tokens=cfg.system_prefix_tokens, producer_node_id=None)
    repo_context = _state("repo_context", "prompt_context", "shared",
                           tokens=cfg.shared_context_tokens, producer_node_id=None)
    emit(v_events.STATE_DECLARE, None, system_prefix)
    emit(v_events.STATE_DECLARE, None, repo_context)

    emit(v_events.STATE_READ, None, {
        "state_id": "system_prefix", "content_hash": "hash_system_prefix_v1",
        "consumer_node_id": planner, "tokens": cfg.system_prefix_tokens,
    })
    emit(v_events.STATE_READ, None, {
        "state_id": "repo_context", "content_hash": "hash_repo_context_v1",
        "consumer_node_id": planner, "tokens": cfg.shared_context_tokens,
    })

    step += 1
    emit("update_status", planner, {"status": "in_progress"})
    step += 1
    emit("update_status", planner, {
        "status": "complete",
        "evidence": ["planner emitted plan"],
    })

    step += 1
    subagents: list[str] = []
    for i in range(cfg.num_subagents):
        sid = f"S{i + 2}"
        label = _subagent_label(i)
        subagents.append(sid)
        emit("add_subtask", sid, {
            "description": f"subagent_{label}",
            "parent_id": planner,
            "weight": 1.0,
            "category": "product",
            "node_type": "subagent",
            "workflow_id": cfg.workflow_id,
            "label": label,
        })

    for i, sid in enumerate(subagents):
        label = _subagent_label(i)
        priv_id = f"private_{label}"
        emit(v_events.STATE_DECLARE, None, _state(
            priv_id, "prompt_context", "private",
            tokens=cfg.private_context_tokens[i], producer_node_id=None,
        ))

    emit(v_events.STATE_DECLARE, None, _state(
        "workspace_AC", "workspace", "shared",
        tokens=0, bytes_=cfg.workspace_bytes, producer_node_id=None,
    ))

    for i, sid in enumerate(subagents):
        label = _subagent_label(i)
        emit(v_events.STATE_READ, None, {
            "state_id": "system_prefix", "content_hash": "hash_system_prefix_v1",
            "consumer_node_id": sid, "tokens": cfg.system_prefix_tokens,
        })
        emit(v_events.STATE_READ, None, {
            "state_id": "repo_context", "content_hash": "hash_repo_context_v1",
            "consumer_node_id": sid, "tokens": cfg.shared_context_tokens,
        })
        emit(v_events.STATE_READ, None, {
            "state_id": f"private_{label}", "content_hash": f"hash_private_{label}_v1",
            "consumer_node_id": sid, "tokens": cfg.private_context_tokens[i],
        })

    if len(subagents) >= 1:
        emit(v_events.STATE_READ, None, {
            "state_id": "workspace_AC", "content_hash": "hash_workspace_AC_v1",
            "consumer_node_id": subagents[0], "tokens": 0,
        })
    if len(subagents) >= 3:
        emit(v_events.STATE_READ, None, {
            "state_id": "workspace_AC", "content_hash": "hash_workspace_AC_v1",
            "consumer_node_id": subagents[2], "tokens": 0,
        })
        emit(v_events.STATE_WRITE, None, {
            "state_id": "workspace_AC", "content_hash": "hash_workspace_AC_v1",
            "producer_node_id": subagents[2], "tokens": 0, "bytes": cfg.workspace_bytes,
        })

    step += 1
    for sid in subagents:
        emit("update_status", sid, {"status": "in_progress"})
    step += 1
    for sid in subagents:
        emit("update_status", sid, {
            "status": "complete",
            "evidence": [f"{sid} returned"],
        })

    return ev


def write_jsonl(event_dicts: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in event_dicts))


def generate_to_file(config: SyntheticConfig, path: str | Path) -> list[dict]:
    ev = generate_events(config)
    write_jsonl(ev, path)
    return ev
