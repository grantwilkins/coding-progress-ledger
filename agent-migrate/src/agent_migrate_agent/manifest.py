"""Serving Group Manifest: replay a agent_migrate trace into a state graph.

Bipartite source_of_truth: nodes <--uses--> state objects.
Pairwise StateEdge is a derived view convenient for policy code.

Edge weight is the token count of the shared state object (NOT the sum of
read.tokens across both consumers — they share the whole object once).
Workspace state has tokens=0; its pairwise edges have weight=0 and survive
in the derived view. Policies filter by tau.

State identity rule:
    primary key = state_id (synthetic adapters supply this)
    secondary  = content_hash (real adapters supply this)
A second state_declare with the same state_id but a different content_hash
hard_fails. MVP does not version state objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from ledger_progress import EventType, Ledger

from . import events as v_events


@dataclass
class WorkNode:
    node_id: str
    node_type: str
    parent_node_id: str | None
    workflow_id: str | None
    label: str | None
    status: str
    required_state: list[str] = field(default_factory=list)
    produced_state: list[str] = field(default_factory=list)
    session_id: str | None = None


@dataclass
class StateObject:
    state_id: str
    content_hash: str
    layer: str
    lifetime: str
    tokens: int
    bytes: int | None
    producers: list[str] = field(default_factory=list)
    consumers: list[str] = field(default_factory=list)
    invalidated: bool = False
    home_site: str | None = None
    role_at_cut: str | None = None


@dataclass(frozen=True)
class StateEdge:
    node_a: str
    node_b: str
    state_id: str
    weight: int


@dataclass
class ServingGroupManifest:
    workflow_id: str | None
    root_task: str
    nodes: dict[str, WorkNode]
    state_objects: dict[str, StateObject]
    edges: list[StateEdge]


def build_manifest(ledger: Ledger) -> ServingGroupManifest:
    nodes: dict[str, WorkNode] = {}
    state_objects: dict[str, StateObject] = {}
    workflow_id: str | None = None

    for event in ledger.events:
        et = event.event_type
        payload = event.payload

        if et is EventType.ADD_SUBTASK:
            wf = payload.get("workflow_id")
            if workflow_id is None:
                workflow_id = wf
            elif wf is not None and wf != workflow_id:
                raise ValueError(f"trace mixes workflow_ids: {workflow_id!r} vs {wf!r}")
            sid = event.subtask_id
            subtask = ledger.subtasks[sid]
            nodes[sid] = WorkNode(
                node_id=sid,
                node_type=payload.get("node_type", "unknown"),
                parent_node_id=subtask.parent_id,
                workflow_id=wf,
                label=payload.get("label"),
                status=subtask.status.value,
                session_id=payload.get("session_id"),
            )
        elif et == v_events.STATE_DECLARE:
            _declare(state_objects, payload)
        elif et == v_events.STATE_READ:
            _read(state_objects, payload)
        elif et == v_events.STATE_WRITE:
            _write(state_objects, payload)
        elif et == v_events.STATE_INVALIDATE:
            sid = payload["state_id"]
            if sid not in state_objects:
                raise ValueError(f"state_invalidate for unknown state_id: {sid}")
            state_objects[sid].invalidated = True

    for sid, node in nodes.items():
        node.status = ledger.subtasks[sid].status.value

    _reconcile_node_state_links(nodes, state_objects)
    _validate_node_references(nodes, state_objects)
    edges = _pairwise_edges(state_objects)
    return ServingGroupManifest(
        workflow_id=workflow_id,
        root_task=ledger.root_task,
        nodes=nodes,
        state_objects=state_objects,
        edges=edges,
    )


def _declare(state_objects: dict[str, StateObject], payload: dict) -> None:
    sid = payload["state_id"]
    if sid in state_objects:
        raise ValueError(
            f"duplicate state_declare for state_id {sid!r}; "
            f"MVP does not version state objects (existing content_hash={state_objects[sid].content_hash!r})"
        )
    producer = payload.get("producer_node_id")
    state_objects[sid] = StateObject(
        state_id=sid,
        content_hash=payload["content_hash"],
        layer=payload["layer"],
        lifetime=payload["lifetime"],
        tokens=int(payload["tokens"]),
        bytes=payload.get("bytes"),
        producers=[producer] if producer else [],
        home_site=payload.get("home_site"),
        role_at_cut=payload.get("role_at_cut"),
    )


def _read(state_objects: dict[str, StateObject], payload: dict) -> None:
    state = _require_state(state_objects, payload["state_id"], payload["content_hash"])
    consumer = payload["consumer_node_id"]
    if consumer not in state.consumers:
        state.consumers.append(consumer)


def _write(state_objects: dict[str, StateObject], payload: dict) -> None:
    state = _require_state(state_objects, payload["state_id"], payload["content_hash"])
    producer = payload["producer_node_id"]
    if producer not in state.producers:
        state.producers.append(producer)
    new_bytes = payload.get("bytes")
    if new_bytes is not None:
        state.bytes = int(new_bytes)


def _reconcile_node_state_links(nodes: dict[str, WorkNode], state_objects: dict[str, StateObject]) -> None:
    """The bipartite source_of_truth lives on StateObject. Mirror it onto WorkNode
    so consumers/policies can read either side without ordering surprises."""
    for state in state_objects.values():
        for consumer in state.consumers:
            if consumer in nodes and state.state_id not in nodes[consumer].required_state:
                nodes[consumer].required_state.append(state.state_id)
        for producer in state.producers:
            if producer in nodes and state.state_id not in nodes[producer].produced_state:
                nodes[producer].produced_state.append(state.state_id)


def _validate_node_references(nodes: dict[str, WorkNode], state_objects: dict[str, StateObject]) -> None:
    """Every producer/consumer node_id referenced by a state object must have an add_subtask event."""
    for state in state_objects.values():
        for ref in state.producers + state.consumers:
            if ref not in nodes:
                raise ValueError(
                    f"state {state.state_id!r} references node {ref!r} that has no add_subtask event"
                )


def _require_state(state_objects: dict[str, StateObject], sid: str, content_hash: str) -> StateObject:
    if sid not in state_objects:
        raise ValueError(f"state_read/write for undeclared state_id: {sid}")
    state = state_objects[sid]
    if state.content_hash != content_hash:
        raise ValueError(
            f"content_hash mismatch on state_id {sid!r}: declared {state.content_hash!r}, "
            f"event referenced {content_hash!r}"
        )
    return state


def _pairwise_edges(state_objects: dict[str, StateObject]) -> list[StateEdge]:
    edges: list[StateEdge] = []
    for state in state_objects.values():
        for a, b in combinations(state.consumers, 2):
            edges.append(StateEdge(node_a=a, node_b=b, state_id=state.state_id, weight=state.tokens))
    return edges
