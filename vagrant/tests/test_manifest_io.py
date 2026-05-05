import json
from pathlib import Path

from ledger_progress import from_jsonl

from vagrant_agent import build_manifest, read_json, to_dict, write_json
from vagrant_agent.manifest_io import from_dict

TRACE_PATH = Path(__file__).resolve().parent.parent / "examples" / "traces" / "toy_subagent_trace.jsonl"


def test_dict_roundtrip():
    m = build_manifest(from_jsonl(str(TRACE_PATH)))
    rebuilt = from_dict(to_dict(m))
    assert rebuilt.workflow_id == m.workflow_id
    assert rebuilt.root_task == m.root_task
    assert set(rebuilt.nodes) == set(m.nodes)
    assert set(rebuilt.state_objects) == set(m.state_objects)
    assert len(rebuilt.edges) == len(m.edges)
    assert {(e.node_a, e.node_b, e.state_id, e.weight) for e in rebuilt.edges} == \
           {(e.node_a, e.node_b, e.state_id, e.weight) for e in m.edges}


def test_json_file_roundtrip(tmp_path: Path):
    m = build_manifest(from_jsonl(str(TRACE_PATH)))
    path = tmp_path / "manifest.json"
    write_json(m, path)
    loaded = read_json(path)
    assert loaded.workflow_id == m.workflow_id
    assert set(loaded.state_objects) == set(m.state_objects)


def test_json_is_valid_and_indented(tmp_path: Path):
    m = build_manifest(from_jsonl(str(TRACE_PATH)))
    path = tmp_path / "manifest.json"
    write_json(m, path)
    text = path.read_text()
    json.loads(text)
    assert "\n  " in text  # indented
