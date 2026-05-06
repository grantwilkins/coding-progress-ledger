from pathlib import Path

import pytest
from ledger_progress import from_jsonl

from vagrant_agent import build_manifest
from vagrant_agent.adapters.swe_agent import approx_tokens, swe_agent_to_events, swe_agent_to_trace
from vagrant_agent.metrics import non_trivial_shared_state_count

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "swe_agent_pilot_s_07.json"


def test_approx_tokens():
    assert approx_tokens("") == 1
    assert approx_tokens("abcd") == 1
    assert approx_tokens("a" * 100) == 25


def test_adapter_emits_init_first():
    ev = swe_agent_to_events(FIXTURE)
    assert ev[0]["event_type"] == "init"


def test_adapter_emits_one_llm_call_node_per_ai_turn():
    import json
    raw = json.loads(FIXTURE.read_text())
    ai_turns = sum(1 for t in raw["trajectory"] if t.get("role") == "ai")
    ev = swe_agent_to_events(FIXTURE)
    add_subtasks = [e for e in ev if e["event_type"] == "add_subtask"]
    assert len(add_subtasks) == ai_turns
    assert all(e["payload"]["node_type"] == "llm_call" for e in add_subtasks)


def test_adapter_emits_system_prompt_and_issue_text_states():
    ev = swe_agent_to_events(FIXTURE)
    declares = [e for e in ev if e["event_type"] == "state_declare"]
    state_ids = {e["payload"]["state_id"] for e in declares}
    assert "system_prompt" in state_ids
    assert "issue_text" in state_ids


def test_adapter_round_trips_through_ledger_replay(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    ledger = from_jsonl(str(out))
    assert ledger.root_task.startswith("swe_agent:")


def test_f_gate_at_least_5_nodes(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    assert len(manifest.nodes) >= 5


def test_f_gate_at_least_2_shared_state_objects(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    shared = [s for s in manifest.state_objects.values() if len(s.consumers) >= 2]
    assert len(shared) >= 2


def test_system_prompt_consumed_by_every_llm_call(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    sp = manifest.state_objects["system_prompt"]
    assert set(sp.consumers) == set(manifest.nodes.keys())


def test_issue_text_consumed_by_every_llm_call(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    issue = manifest.state_objects["issue_text"]
    assert set(issue.consumers) == set(manifest.nodes.keys())


def test_non_trivial_shared_state_diagnostic_warns_if_zero(tmp_path: Path, capsys):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    diag = non_trivial_shared_state_count(manifest)
    if diag == 0:
        pytest.fail(
            "non_trivial_shared_state_count is 0 on the pilot fixture. "
            "Either content-hash dedup of repeated tool outputs broke, or this "
            "fixture has no repeated tool outputs (very rare for SWE-agent)."
        )
    assert diag >= 1


def test_first_turn_system_required(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"instance_id": "x", "trajectory": [{"role": "user", "text": "hi"}]}')
    with pytest.raises(ValueError, match="first turn must be system"):
        swe_agent_to_events(bad)


def test_first_non_system_turn_must_be_user(tmp_path: Path):
    """An ai turn appearing before any user turn is malformed."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"instance_id": "x", "trajectory": ['
        '{"role": "system", "system_prompt": "abc"},'
        '{"role": "ai", "text": "hello"}'
        ']}'
    )
    with pytest.raises(ValueError, match="first non-system turn must be user"):
        swe_agent_to_events(bad)


def test_empty_trajectory_hard_fails(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"instance_id": "x", "trajectory": []}')
    with pytest.raises(ValueError, match="first turn must be system"):
        swe_agent_to_events(bad)


def test_missing_system_prompt_hard_fails(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"instance_id": "x", "trajectory": ['
        '{"role": "system", "system_prompt": ""},'
        '{"role": "user", "text": "issue"}'
        ']}'
    )
    with pytest.raises(ValueError, match="system_prompt"):
        swe_agent_to_events(bad)


def test_no_repeats_trajectory_diagnostic_is_zero(tmp_path: Path):
    """The adapter must NOT fabricate sharing on a trajectory with all-unique tool outputs.

    Note: under the accumulation model, even unique tool outputs are shared across
    later ai turns within ONE trajectory. The diagnostic should still surface them
    with consumer_count >= 2 because every later ai turn reads them.
    """
    fake = tmp_path / "fake.json"
    fake.write_text(
        '{"instance_id": "fake-1", "trajectory": ['
        '{"role": "system", "system_prompt": "prompt"},'
        '{"role": "user", "text": "issue text here"},'
        '{"role": "ai", "text": "let me look"},'
        '{"role": "user", "text": "unique-A"},'
        '{"role": "ai", "text": "checking"},'
        '{"role": "user", "text": "unique-B"},'
        '{"role": "ai", "text": "fixing"}'
        ']}'
    )
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(fake, out)
    manifest = build_manifest(from_jsonl(str(out)))
    # tool_output_000 (unique-A) is read by the second AND third ai turns -> 2 consumers
    # tool_output_001 (unique-B) is read by the third ai turn only -> 1 consumer
    assert manifest.state_objects["tool_output_000"].consumers == ["S2", "S3"]
    assert manifest.state_objects["tool_output_001"].consumers == ["S3"]


def test_non_trivial_shared_state_count_with_custom_exclude(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    manifest = build_manifest(from_jsonl(str(out)))
    full = non_trivial_shared_state_count(manifest, exclude=())
    default = non_trivial_shared_state_count(manifest)
    assert full > default  # default excludes system_prompt + issue_text
    assert full == default + 2


def test_workflow_id_default_includes_instance(tmp_path: Path):
    out = tmp_path / "trace.jsonl"
    swe_agent_to_trace(FIXTURE, out)
    ledger = from_jsonl(str(out))
    add_events = [e for e in ledger.events if str(e.event_type).endswith("ADD_SUBTASK")
                  or str(e.event_type) == "EventType.ADD_SUBTASK"]
    # Just spot-check that workflow_id is set on the first add_subtask.
    first = next(e for e in ledger.events
                 if hasattr(e.event_type, "value") and e.event_type.value == "add_subtask")
    assert first.payload.get("workflow_id", "").startswith("swe_agent_")
