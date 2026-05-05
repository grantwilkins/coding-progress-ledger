from pathlib import Path

import pytest
from ledger_progress import EventType, LedgerEvent, LedgerSession, apply_event, from_jsonl, to_jsonl

from vagrant_agent import build_manifest
from vagrant_agent import events as v_events
from vagrant_agent.adapters.synthetic import SyntheticConfig, generate_to_file

TRACE_PATH = Path(__file__).resolve().parent.parent / "examples" / "traces" / "toy_subagent_trace.jsonl"


def _toy_manifest():
    return build_manifest(from_jsonl(str(TRACE_PATH)))


def test_node_count_and_types():
    m = _toy_manifest()
    assert set(m.nodes) == {"S1", "S2", "S3", "S4"}
    assert m.nodes["S1"].node_type == "llm_call"
    assert all(m.nodes[s].node_type == "subagent" for s in ["S2", "S3", "S4"])


def test_workflow_id_recovered():
    m = _toy_manifest()
    assert m.workflow_id == "toy_workflow_v1"
    assert m.root_task == "toy coding task"


def test_node_parent_links():
    m = _toy_manifest()
    assert m.nodes["S1"].parent_node_id is None
    assert m.nodes["S2"].parent_node_id == "S1"
    assert m.nodes["S3"].parent_node_id == "S1"
    assert m.nodes["S4"].parent_node_id == "S1"


def test_node_status_is_final():
    m = _toy_manifest()
    assert all(node.status == "complete" for node in m.nodes.values())


def test_state_object_count():
    m = _toy_manifest()
    assert set(m.state_objects) == {
        "system_prefix", "repo_context", "private_A", "private_B", "private_C", "workspace_AC",
    }


def test_shared_state_consumer_counts():
    m = _toy_manifest()
    assert m.state_objects["system_prefix"].consumers == ["S1", "S2", "S3", "S4"]
    assert m.state_objects["repo_context"].consumers == ["S1", "S2", "S3", "S4"]


def test_workspace_consumers_and_producer():
    m = _toy_manifest()
    ws = m.state_objects["workspace_AC"]
    assert ws.consumers == ["S2", "S4"]
    assert ws.producers == ["S4"]
    assert ws.bytes == 4_000_000
    assert ws.tokens == 0


def test_private_state_isolation():
    m = _toy_manifest()
    assert m.state_objects["private_A"].consumers == ["S2"]
    assert m.state_objects["private_B"].consumers == ["S3"]
    assert m.state_objects["private_C"].consumers == ["S4"]


def test_required_state_per_node():
    m = _toy_manifest()
    assert set(m.nodes["S1"].required_state) == {"system_prefix", "repo_context"}
    assert set(m.nodes["S2"].required_state) == {"system_prefix", "repo_context", "private_A", "workspace_AC"}
    assert set(m.nodes["S3"].required_state) == {"system_prefix", "repo_context", "private_B"}
    assert set(m.nodes["S4"].required_state) == {"system_prefix", "repo_context", "private_C", "workspace_AC"}


def test_produced_state():
    m = _toy_manifest()
    assert m.nodes["S4"].produced_state == ["workspace_AC"]
    for n in ["S1", "S2", "S3"]:
        assert m.nodes[n].produced_state == []


def test_pairwise_edge_counts():
    m = _toy_manifest()
    expected_pairs = {
        "system_prefix": 6,   # C(4,2)
        "repo_context": 6,    # C(4,2)
        "workspace_AC": 1,    # C(2,2)
        "private_A": 0,
        "private_B": 0,
        "private_C": 0,
    }
    by_state: dict[str, int] = {}
    for edge in m.edges:
        by_state[edge.state_id] = by_state.get(edge.state_id, 0) + 1
    for sid, expected in expected_pairs.items():
        assert by_state.get(sid, 0) == expected, f"{sid}: expected {expected}, got {by_state.get(sid, 0)}"
    assert len(m.edges) == 13


def test_edge_weights_use_state_tokens():
    m = _toy_manifest()
    repo_edges = [e for e in m.edges if e.state_id == "repo_context"]
    assert all(e.weight == 8000 for e in repo_edges)
    sys_edges = [e for e in m.edges if e.state_id == "system_prefix"]
    assert all(e.weight == 200 for e in sys_edges)
    ws_edges = [e for e in m.edges if e.state_id == "workspace_AC"]
    assert len(ws_edges) == 1 and ws_edges[0].weight == 0


def test_edges_are_undirected_unique_pairs():
    m = _toy_manifest()
    seen = set()
    for e in m.edges:
        key = (frozenset({e.node_a, e.node_b}), e.state_id)
        assert key not in seen, f"duplicate edge {key}"
        assert e.node_a != e.node_b
        seen.add(key)


# ---- error paths ----

def test_state_id_collision_with_different_hash_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "hA", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "hB", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    ledger = from_jsonl(str(path))
    with pytest.raises(ValueError, match="duplicate state_declare"):
        build_manifest(ledger)


def test_state_read_for_undeclared_state_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_READ, "subtask_id": None,
         "payload": {"state_id": "ghost", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": 1}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    ledger = from_jsonl(str(path))
    with pytest.raises(ValueError, match="undeclared"):
        build_manifest(ledger)


def test_content_hash_mismatch_on_read_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "hA", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_READ, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "hB",
                     "consumer_node_id": "S1", "tokens": 10}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    ledger = from_jsonl(str(path))
    with pytest.raises(ValueError, match="content_hash mismatch"):
        build_manifest(ledger)


def test_state_declare_with_producer_mirrors_to_node(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "art", "content_hash": "h",
                     "layer": "workspace", "lifetime": "shared",
                     "tokens": 0, "bytes": 1000,
                     "producer_node_id": "S1"}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    m = build_manifest(from_jsonl(str(path)))
    assert m.state_objects["art"].producers == ["S1"]
    assert m.nodes["S1"].produced_state == ["art"]


def test_state_invalidate_marks_object(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "hA", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": v_events.STATE_INVALIDATE, "subtask_id": None,
         "payload": {"state_id": "x", "reason": "stale"}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    m = build_manifest(from_jsonl(str(path)))
    assert m.state_objects["x"].invalidated is True


def test_state_write_for_undeclared_state_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_WRITE, "subtask_id": None,
         "payload": {"state_id": "ghost", "content_hash": "h",
                     "producer_node_id": "S1", "tokens": 0,
                     "bytes": 100}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    with pytest.raises(ValueError, match="undeclared"):
        build_manifest(from_jsonl(str(path)))


def test_duplicate_state_declare_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 2, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    with pytest.raises(ValueError, match="duplicate state_declare"):
        build_manifest(from_jsonl(str(path)))


def test_consumer_node_never_declared_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 10, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_READ, "subtask_id": None,
         "payload": {"state_id": "x", "content_hash": "h",
                     "consumer_node_id": "Sphantom", "tokens": 10}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    with pytest.raises(ValueError, match="Sphantom"):
        build_manifest(from_jsonl(str(path)))


def test_state_event_before_add_subtask_still_reconciles(tmp_path: Path):
    """Out-of-order: state_declare/read fires before the consumer's add_subtask.
    The reconciliation pass must still populate node.required_state."""
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_DECLARE, "subtask_id": None,
         "payload": {"state_id": "ctx", "content_hash": "h", "layer": "prompt_context",
                     "lifetime": "shared", "tokens": 100, "bytes": None,
                     "producer_node_id": None}, "reason": None},
        {"step": 1, "event_type": v_events.STATE_READ, "subtask_id": None,
         "payload": {"state_id": "ctx", "content_hash": "h",
                     "consumer_node_id": "S1", "tokens": 100}, "reason": None},
        {"step": 2, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call"}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    m = build_manifest(from_jsonl(str(path)))
    assert m.state_objects["ctx"].consumers == ["S1"]
    assert m.nodes["S1"].required_state == ["ctx"]


def test_workstream_b_gate_consumer_counts():
    """The TASKS.md gate: shared=4, workspace=2, private=1 each, all in one assertion block."""
    m = _toy_manifest()
    counts = {sid: len(s.consumers) for sid, s in m.state_objects.items()}
    assert counts == {
        "system_prefix": 4,
        "repo_context": 4,
        "workspace_AC": 2,
        "private_A": 1,
        "private_B": 1,
        "private_C": 1,
    }


def test_mixed_workflow_ids_hard_fails(tmp_path: Path):
    from vagrant_agent.adapters.synthetic import write_jsonl

    events = [
        {"step": 0, "event_type": "init", "subtask_id": None,
         "payload": {"root_task": "r"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S1",
         "payload": {"description": "n", "parent_id": None, "weight": 1.0,
                     "category": "product", "node_type": "llm_call",
                     "workflow_id": "wf_a"}, "reason": None},
        {"step": 1, "event_type": "add_subtask", "subtask_id": "S2",
         "payload": {"description": "n", "parent_id": "S1", "weight": 1.0,
                     "category": "product", "node_type": "subagent",
                     "workflow_id": "wf_b"}, "reason": None},
    ]
    path = tmp_path / "trace.jsonl"
    write_jsonl(events, path)
    ledger = from_jsonl(str(path))
    with pytest.raises(ValueError, match="workflow_id"):
        build_manifest(ledger)
