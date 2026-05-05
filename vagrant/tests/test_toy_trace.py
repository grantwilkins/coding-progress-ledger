from pathlib import Path

from vagrant_agent.adapters.synthetic import SyntheticConfig, generate_events, write_jsonl

TRACE_PATH = Path(__file__).resolve().parent.parent / "examples" / "traces" / "toy_subagent_trace.jsonl"


def test_committed_trace_matches_generator(tmp_path: Path):
    ev = generate_events(SyntheticConfig())
    expected = tmp_path / "expected.jsonl"
    write_jsonl(ev, expected)
    assert TRACE_PATH.read_bytes() == expected.read_bytes()


def test_committed_trace_event_count():
    lines = TRACE_PATH.read_text().splitlines()
    assert len(lines) == 33
