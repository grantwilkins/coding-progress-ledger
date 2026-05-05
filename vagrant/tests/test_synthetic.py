import json
from pathlib import Path

import pytest

from vagrant_agent import events as v_events
from vagrant_agent.adapters.synthetic import SyntheticConfig, generate_events, generate_to_file, write_jsonl


def test_default_config_consumer_counts():
    ev = generate_events(SyntheticConfig())
    reads = [e for e in ev if e["event_type"] == v_events.STATE_READ]
    by_state: dict[str, set[str]] = {}
    for e in reads:
        by_state.setdefault(e["payload"]["state_id"], set()).add(e["payload"]["consumer_node_id"])
    assert by_state["system_prefix"] == {"S1", "S2", "S3", "S4"}
    assert by_state["repo_context"] == {"S1", "S2", "S3", "S4"}
    assert by_state["workspace_AC"] == {"S2", "S4"}
    assert by_state["private_A"] == {"S2"}
    assert by_state["private_B"] == {"S3"}
    assert by_state["private_C"] == {"S4"}


def test_writes_one_workspace():
    ev = generate_events(SyntheticConfig())
    writes = [e for e in ev if e["event_type"] == v_events.STATE_WRITE]
    assert len(writes) == 1
    assert writes[0]["payload"]["state_id"] == "workspace_AC"
    assert writes[0]["payload"]["producer_node_id"] == "S4"


def test_first_event_is_init():
    ev = generate_events(SyntheticConfig())
    assert ev[0]["event_type"] == "init"


def test_subagent_count_matches_private_tokens():
    with pytest.raises(ValueError):
        SyntheticConfig(num_subagents=2, private_context_tokens=(1, 2, 3))


def test_deterministic(tmp_path: Path):
    cfg = SyntheticConfig(seed=42)
    a = generate_to_file(cfg, tmp_path / "a.jsonl")
    b = generate_to_file(cfg, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert a == b


def test_jsonl_roundtrip(tmp_path: Path):
    ev = generate_events(SyntheticConfig())
    path = tmp_path / "trace.jsonl"
    write_jsonl(ev, path)
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == ev


def test_node_types_present():
    ev = generate_events(SyntheticConfig())
    adds = [e for e in ev if e["event_type"] == "add_subtask"]
    types = {e["payload"]["node_type"] for e in adds}
    assert types == {"llm_call", "subagent"}
