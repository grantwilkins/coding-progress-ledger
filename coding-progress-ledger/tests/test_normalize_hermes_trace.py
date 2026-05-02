"""Hermes normalizer tests (HP2). Mirrors test_normalize_swe_agent_trace.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalize_hermes_trace import (
    SCHEMA_VERSION,
    main,
    normalize_row,
    render_summary,
)


SAMPLE_KIMI = (
    Path(__file__).resolve().parents[1] / "external_data" / "hermes" / "raw" / "sample_row_kimi.json"
)


def _system(prompt):
    return {"from": "system", "value": prompt}


def _human(text):
    return {"from": "human", "value": text}


def _gpt(value):
    return {"from": "gpt", "value": value}


def _tool(value):
    return {"from": "tool", "value": value}


def test_top_level_final_success_always_none_even_with_extra_keys():
    row = {"id": "x", "conversations": [], "category": "Terminal & Coding", "target": True, "resolved": True}
    norm = normalize_row(row, model_name="kimi")
    assert norm["final_success"] is None
    assert norm["exit_status"] is None


def test_role_mapping_system_human_gpt_tool():
    row = {
        "id": "x",
        "conversations": [
            _system("sp"),
            _human("hi"),
            _gpt("just a thought"),
            _tool("<tool_response>\n{\"tool_call_id\":\"a\",\"name\":\"t\",\"content\":\"ok\"}\n</tool_response>"),
        ],
    }
    norm = normalize_row(row, model_name="kimi")
    roles = [ev["role"] for ev in norm["events"]]
    assert roles == ["system", "user", "assistant", "tool"]


def test_think_blocks_collapsed_into_thought_field():
    text = "<think>\ninner reasoning\n</think>\nThe answer."
    row = {"id": "x", "conversations": [_gpt(text)]}
    norm = normalize_row(row, model_name="kimi")
    ev = norm["events"][0]
    assert ev["thought"] is not None
    assert "inner reasoning" in ev["thought"]
    assert "<think>" not in (ev["thought"] or "")
    assert "The answer." in ev["thought"]


def test_thought_only_turn_kept_as_step_with_empty_action():
    row = {"id": "x", "conversations": [_gpt("<think>thinking</think>\nNo tool call here.")]}
    norm = normalize_row(row, model_name="kimi")
    assert len(norm["events"]) == 1
    ev = norm["events"][0]
    assert ev["role"] == "assistant"
    assert ev["action"] is None
    assert ev["command"] is None
    assert ev["tool_name"] is None


def test_multi_tool_call_turn_splits_into_one_step_per_call():
    value = """<think>plan</think>
Search for stuff:
<tool_call>
{"name": "search_files", "arguments": {"target": "files", "pattern": "*.yaml"}}
</tool_call>
<tool_call>
{"name": "search_files", "arguments": {"target": "files", "pattern": "*.yml"}}
</tool_call>
<tool_call>
{"name": "terminal", "arguments": {"cmd": "ls"}}
</tool_call>
<tool_call>
{"name": "search_files", "arguments": {"target": "files", "pattern": "*staging*"}}
</tool_call>"""
    row = {"id": "x", "conversations": [_gpt(value)]}
    norm = normalize_row(row, model_name="kimi")
    assistants = [ev for ev in norm["events"] if ev["role"] == "assistant"]
    assert len(assistants) == 4
    assert [ev["tool_name"] for ev in assistants] == ["search_files", "search_files", "terminal", "search_files"]


def test_free_text_thought_attaches_to_first_split_step_only():
    value = """<think>plan</think>
Free text response.
<tool_call>
{"name": "a", "arguments": {}}
</tool_call>
<tool_call>
{"name": "b", "arguments": {}}
</tool_call>"""
    row = {"id": "x", "conversations": [_gpt(value)]}
    norm = normalize_row(row, model_name="kimi")
    asst = [ev for ev in norm["events"] if ev["role"] == "assistant"]
    assert asst[0]["thought"] is not None
    assert "Free text response" in asst[0]["thought"]
    assert asst[1]["thought"] is None


def test_tool_call_id_pairs_call_to_response_when_ids_present():
    gpt_val = """<tool_call>
{"id": "call_a", "name": "search_files", "arguments": {"q": "x"}}
</tool_call>
<tool_call>
{"id": "call_b", "name": "terminal", "arguments": {"cmd": "ls"}}
</tool_call>"""
    tool_val = """<tool_response>
{"tool_call_id": "call_b", "name": "terminal", "content": "B response"}
</tool_response>
<tool_response>
{"tool_call_id": "call_a", "name": "search_files", "content": "A response"}
</tool_response>"""
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    norm = normalize_row(row, model_name="kimi")
    tools = [ev for ev in norm["events"] if ev["role"] == "tool"]
    assert len(tools) == 2
    assert tools[0]["tool_name"] == "search_files"
    assert tools[0]["observation"] == "A response"
    assert tools[1]["tool_name"] == "terminal"
    assert tools[1]["observation"] == "B response"


def test_tool_response_pairs_positionally_when_ids_missing():
    gpt_val = """<tool_call>
{"name": "a", "arguments": {}}
</tool_call>
<tool_call>
{"name": "b", "arguments": {}}
</tool_call>"""
    tool_val = """<tool_response>
{"tool_call_id": "irrelevant", "name": "a", "content": "first"}
</tool_response>
<tool_response>
{"tool_call_id": "irrelevant2", "name": "b", "content": "second"}
</tool_response>"""
    row = {"id": "x", "conversations": [_gpt(gpt_val), _tool(tool_val)]}
    norm = normalize_row(row, model_name="kimi")
    tools = [ev for ev in norm["events"] if ev["role"] == "tool"]
    assert tools[0]["observation"] == "first"
    assert tools[1]["observation"] == "second"


def test_schema_version_and_source():
    norm = normalize_row({"id": "x", "conversations": []}, model_name="kimi")
    assert norm["schema_version"] == SCHEMA_VERSION
    assert norm["source"] == "hermes_agent_reasoning"


def test_non_dict_input_raises():
    with pytest.raises(TypeError):
        normalize_row(["nope"])


def test_non_list_conversations_raises():
    with pytest.raises(TypeError):
        normalize_row({"conversations": "oops"})


def test_files_touched_for_path_tools():
    val = """<tool_call>
{"name": "write_file", "arguments": {"path": "/repo/foo.py", "content": "x"}}
</tool_call>"""
    norm = normalize_row({"id": "x", "conversations": [_gpt(val)]}, model_name="kimi")
    assert norm["events"][0]["files_touched"] == ["/repo/foo.py"]


def test_raw_metadata_includes_category_and_subcategory():
    row = {"id": "x", "conversations": [], "category": "Terminal & Coding", "subcategory": "Build"}
    norm = normalize_row(row, model_name="kimi")
    assert norm["raw_metadata"]["category"] == "Terminal & Coding"
    assert norm["raw_metadata"]["subcategory"] == "Build"


def test_step_index_dense_zero_based():
    row = {"id": "x", "conversations": [_system("sp"), _human("hi"), _gpt("thought")]}
    norm = normalize_row(row, model_name="kimi")
    assert [ev["step_index"] for ev in norm["events"]] == [0, 1, 2]


@pytest.mark.skipif(not SAMPLE_KIMI.exists(), reason="sample row missing")
def test_kimi_sample_row_post_split_step_count():
    row = json.loads(SAMPLE_KIMI.read_text(encoding="utf-8"))
    norm = normalize_row(row, model_name="kimi")
    # Pre-split conversation count is 13. The kimi sample has multi-tool-call gpt turns,
    # so the post-split count must be strictly greater than the pre-split count.
    assert len(norm["events"]) > len(row["conversations"])
    assert norm["final_success"] is None
    assert norm["instance_id"] == row["id"]
    assert norm["model_name"] == "kimi"


def test_main_writes_artifacts(tmp_path):
    src = tmp_path / "row.json"
    src.write_text(json.dumps({
        "id": "abc",
        "category": "Terminal & Coding",
        "subcategory": "x",
        "conversations": [
            _system("sp"),
            _human("issue"),
            _gpt("<think>r</think>\n<tool_call>\n{\"name\": \"ls\", \"arguments\": {}}\n</tool_call>"),
            _tool("<tool_response>\n{\"tool_call_id\": \"a\", \"name\": \"ls\", \"content\": \"ok\"}\n</tool_response>"),
        ],
    }), encoding="utf-8")
    run_dir = tmp_path / "run"
    rc = main(["--source-row", str(src), "--run-dir", str(run_dir), "--model-name", "kimi"])
    assert rc == 0
    nt = json.loads((run_dir / "normalized_trace.json").read_text(encoding="utf-8"))
    assert nt["instance_id"] == "abc"
    assert nt["final_success"] is None
    summary = (run_dir / "trajectory_summary.md").read_text(encoding="utf-8")
    assert "abc" in summary
