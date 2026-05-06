from pathlib import Path

from ledger_progress import from_jsonl

from agent_migrate_agent import events as v_events
from agent_migrate_agent.adapters.synthetic import SyntheticConfig, generate_events, write_jsonl

TRACE_PATH = Path(__file__).resolve().parent.parent / "examples" / "traces" / "toy_subagent_trace.jsonl"


def test_committed_trace_matches_generator(tmp_path: Path):
    ev = generate_events(SyntheticConfig())
    expected = tmp_path / "expected.jsonl"
    write_jsonl(ev, expected)
    assert TRACE_PATH.read_bytes() == expected.read_bytes()


def test_committed_trace_event_count():
    lines = TRACE_PATH.read_text().splitlines()
    assert len(lines) == 33


def test_committed_trace_replays_under_ledger_progress():
    ledger = from_jsonl(str(TRACE_PATH))
    assert ledger.root_task == "toy coding task"
    expected_ids = {"S1", "S2", "S3", "S4"}
    assert set(ledger.subtasks) == expected_ids
    assert all(st.status.value == "complete" for st in ledger.subtasks.values())


def test_committed_trace_preserves_agent_migrate_events():
    ledger = from_jsonl(str(TRACE_PATH))
    types = [e.event_type for e in ledger.events]
    agent_migrate_event_strings = [t for t in types if isinstance(t, str) and t in v_events.ALL]
    state_reads = [t for t in agent_migrate_event_strings if t == v_events.STATE_READ]
    assert len(state_reads) >= 12  # 4 nodes x 2 shared + 3 private + 2 workspace reads
