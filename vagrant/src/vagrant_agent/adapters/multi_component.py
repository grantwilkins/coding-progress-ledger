"""Multi-component synthetic fixture for Workstreams G/H demonstrations.

The single-component toy demonstrates D1 vs D2. To exercise G1/G2 (which can
only differ from D2 when independent per-component placement decisions cause
cross-component duplication), we need a fixture where:

  - Two strongly-anchored components (each pinned to a different site by a
    private state with `home_site` set).
  - One cross-component state shared by all consumers, anchored at one of the
    sites.

D2's per-component min-cost placement will independently send each component
to its private state's home site, causing the cross-component state to
duplicate. G1's brute-force enumeration sees the duplication and may force
co-location.

Topology:

    S1 ──reads──> private_X (home=phoenix, large)
    S2 ──reads──> private_X
    S3 ──reads──> private_Y (home=seattle, large)
    S4 ──reads──> private_Y
    S1, S2, S3, S4 all read shared_xc (home=phoenix, medium) and tiny_prefix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import events as v_events


@dataclass(frozen=True)
class MultiComponentConfig:
    private_x_tokens: int = 12000
    private_y_tokens: int = 12000
    private_x_home: str = "phoenix"
    private_y_home: str = "seattle"
    shared_xc_tokens: int = 4000
    shared_xc_home: str = "phoenix"
    tiny_prefix_tokens: int = 100
    workflow_id: str = "g_demo_workflow_v1"
    root_task: str = "G demo: two components with cross-component shared state"


def generate_events(config: MultiComponentConfig) -> list[dict]:
    cfg = config
    ev: list[dict] = []
    step = 0

    def emit(event_type: str, subtask_id: str | None, payload: dict, reason: str | None = None) -> None:
        nonlocal step
        ev.append({"step": step, "event_type": event_type, "subtask_id": subtask_id,
                   "payload": payload, "reason": reason})

    emit("init", None, {"root_task": cfg.root_task})
    step += 1

    nodes = ["S1", "S2", "S3", "S4"]
    for sid in nodes:
        emit("add_subtask", sid, {
            "description": f"node_{sid}",
            "parent_id": None,
            "weight": 1.0,
            "category": "product",
            "node_type": "llm_call",
            "workflow_id": cfg.workflow_id,
        })

    states = [
        ("tiny_prefix", "prompt_context", "shared", cfg.tiny_prefix_tokens, None, None),
        ("private_X", "prompt_context", "shared", cfg.private_x_tokens, None, cfg.private_x_home),
        ("private_Y", "prompt_context", "shared", cfg.private_y_tokens, None, cfg.private_y_home),
        ("shared_xc", "prompt_context", "shared", cfg.shared_xc_tokens, None, cfg.shared_xc_home),
    ]
    for state_id, layer, lifetime, tokens, bytes_, home in states:
        emit(v_events.STATE_DECLARE, None, {
            "state_id": state_id,
            "content_hash": f"hash_{state_id}_v1",
            "layer": layer,
            "lifetime": lifetime,
            "tokens": tokens,
            "bytes": bytes_,
            "producer_node_id": None,
            "home_site": home,
        })

    consumption = {
        "tiny_prefix": ["S1", "S2", "S3", "S4"],
        "shared_xc": ["S1", "S2", "S3", "S4"],
        "private_X": ["S1", "S2"],
        "private_Y": ["S3", "S4"],
    }
    for state_id in ("tiny_prefix", "shared_xc", "private_X", "private_Y"):
        meta = next(s for s in states if s[0] == state_id)
        for consumer in consumption[state_id]:
            emit(v_events.STATE_READ, None, {
                "state_id": state_id,
                "content_hash": f"hash_{state_id}_v1",
                "consumer_node_id": consumer,
                "tokens": meta[3],
            })

    step += 1
    for sid in nodes:
        emit("update_status", sid, {"status": "complete", "evidence": [f"{sid} done"]})

    return ev


def generate_to_file(config: MultiComponentConfig, path: str | Path) -> list[dict]:
    ev = generate_events(config)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in ev))
    return ev
