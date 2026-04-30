"""
Claim:
normalize_swe_agent_trace.py converts one upstream nebius row into a
C1-schema-conformant normalized trace: per-step thought/command split
at fenced code blocks, role mapping (system / first-user→environment /
later-user→tool / ai→assistant), final_success mirrored from `target`
preserving the bool-vs-missing distinction, and every upstream field
preserved under raw / raw_metadata.

Plausible wrong implementations:
- Assistant-turn parser splits on the first "```" only, so the closing
  fence is missed and trailing prose leaks into `command`.
- Every `user` row is mapped to `tool`, so the issue text is filed as
  a tool return and `issue_text` is None.
- `final_success` derived via truthiness, so target=False degrades to
  "missing" (or target=None degrades to False).
- `tool_name` on a tool turn taken from the tool text's first token
  rather than from the preceding assistant command.
- Upstream-only keys on a trajectory entry (e.g. mask, cutoff_date,
  system_prompt) get dropped from `raw`.
- `trajectory_length` computed from upstream length rather than from
  the actual events list.
- Unrecognized top-level row keys silently dropped (no audit trail).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.normalize_swe_agent_trace import (
    SCHEMA_VERSION,
    main,
    normalize_row,
    render_summary,
)


# ---------- assistant turn parsing ----------


def _ai(text):
    return {"role": "ai", "text": text, "system_prompt": None, "mask": True, "cutoff_date": None}


def _user(text):
    return {"role": "user", "text": text, "system_prompt": None, "mask": False, "cutoff_date": None}


def _system(prompt):
    return {"role": "system", "text": None, "system_prompt": prompt, "mask": False, "cutoff_date": "01.01.2023"}


def test_assistant_turn_splits_on_fence_pair_not_just_open_fence():
    text = "Reasoning paragraph.\n\n```\nfind_file memset.py lexicon\n```\nTrailing prose that must NOT leak into command."
    row = {"trajectory": [_ai(text)]}
    norm = normalize_row(row)
    ev = norm["events"][0]
    assert ev["role"] == "assistant"
    assert ev["thought"] == "Reasoning paragraph."
    assert ev["command"] == "find_file memset.py lexicon"
    assert "Trailing prose" not in (ev["command"] or "")


def test_assistant_turn_with_no_fence_is_thought_only_and_records_warning():
    row = {"trajectory": [_ai("just thinking, no command")]}
    norm = normalize_row(row)
    ev = norm["events"][0]
    assert ev["thought"] == "just thinking, no command"
    assert ev["command"] is None
    assert ev["action"] is None
    assert ev["tool_name"] is None
    assert "no_fenced_block" in ev["raw"]["parse_warnings"]


def test_assistant_turn_with_text_none_records_text_is_none_warning():
    row = {"trajectory": [_ai(None)]}
    norm = normalize_row(row)
    ev = norm["events"][0]
    assert ev["thought"] is None
    assert ev["command"] is None
    assert "text_is_none" in ev["raw"]["parse_warnings"]


def test_assistant_turn_with_unterminated_fence_records_warning_and_keeps_thought():
    text = "leading thought\n```\nfind_file foo"  # no closing fence
    row = {"trajectory": [_ai(text)]}
    norm = normalize_row(row)
    ev = norm["events"][0]
    assert ev["thought"] == "leading thought"
    assert ev["command"] is None
    assert "fence_unterminated" in ev["raw"]["parse_warnings"]


def test_tool_name_extracted_from_command_first_whitespace_token():
    row = {"trajectory": [_ai("ok\n```\nfind_file memset.py lexicon\n```")]}
    norm = normalize_row(row)
    assert norm["events"][0]["tool_name"] == "find_file"


# ---------- role mapping ----------


def test_first_user_after_system_becomes_environment_carrying_issue_text():
    row = {
        "trajectory": [
            _system("SETTING: programmer..."),
            _user("ISSUE:\nMemset provider crash..."),
            _ai("plan\n```\nls\n```"),
            _user("file1\nfile2\n"),
        ],
    }
    norm = normalize_row(row)
    roles = [ev["role"] for ev in norm["events"]]
    assert roles == ["system", "environment", "assistant", "tool"]
    assert norm["issue_text"].startswith("ISSUE:")
    # The issue text lives only in the environment event's observation,
    # not duplicated into the assistant or tool events.
    assert norm["events"][1]["observation"].startswith("ISSUE:")
    assert norm["events"][2]["observation"] is None
    assert norm["events"][3]["observation"] == "file1\nfile2\n"


def test_subsequent_user_returns_become_tool_with_tool_name_from_prior_assistant():
    # Assistant says `find_file lexicon`; tool returns `Found 1 matches`.
    # Buggy impls often take tool_name from the tool's own text.
    row = {
        "trajectory": [
            _system("sp"),
            _user("issue body"),
            _ai("step\n```\nfind_file lexicon\n```"),
            _user("Found 1 matches in /repo:\n/repo/lexicon/memset.py"),
            _ai("step2\n```\nopen lexicon/memset.py\n```"),
            _user("(content)\n"),
        ],
    }
    norm = normalize_row(row)
    tool_events = [ev for ev in norm["events"] if ev["role"] == "tool"]
    assert len(tool_events) == 2
    assert tool_events[0]["tool_name"] == "find_file"
    assert tool_events[1]["tool_name"] == "open"
    # tool_name must NOT be derived from the tool's own text.
    assert tool_events[0]["tool_name"] != "Found"


def test_unknown_upstream_role_is_preserved_under_raw_and_marked_unknown():
    row = {"trajectory": [{"role": "moderator", "text": "hi"}]}
    norm = normalize_row(row)
    ev = norm["events"][0]
    assert ev["role"] == "unknown"
    assert ev["raw"]["role"] == "moderator"


def test_assistant_with_no_command_nulls_out_tool_name_for_following_tool():
    row = {
        "trajectory": [
            _system("sp"),
            _user("issue"),
            _ai("just thinking, no command"),
            _user("env reply"),
        ],
    }
    norm = normalize_row(row)
    tool_ev = norm["events"][3]
    assert tool_ev["role"] == "tool"
    assert tool_ev["tool_name"] is None


# ---------- final_success from `target` ----------


def test_target_false_yields_final_success_false_not_null():
    norm = normalize_row({"target": False, "trajectory": []})
    assert norm["final_success"] is False


def test_target_true_yields_final_success_true():
    norm = normalize_row({"target": True, "trajectory": []})
    assert norm["final_success"] is True


def test_target_missing_yields_final_success_null_not_false():
    norm = normalize_row({"trajectory": []})
    assert norm["final_success"] is None


def test_target_non_bool_yields_final_success_null_not_truthy_value():
    # Defensive: a row with target=1 (non-bool int) must not be coerced.
    norm = normalize_row({"target": 1, "trajectory": []})
    assert norm["final_success"] is None


# ---------- tolerance and raw preservation ----------


def test_raw_event_preserves_all_upstream_keys_including_unknown_ones():
    upstream = {"role": "ai", "text": "t\n```\nls\n```", "mask": True, "cutoff_date": "01.01.2023", "weird_new_key": 42}
    row = {"trajectory": [upstream]}
    ev = normalize_row(row)["events"][0]
    for k, v in upstream.items():
        assert ev["raw"][k] == v


def test_unknown_top_level_keys_recorded_in_raw_metadata():
    row = {"instance_id": "a", "trajectory": [], "weird_top": "x", "another_extra": [1, 2]}
    norm = normalize_row(row)
    assert norm["raw_metadata"]["extra_top_level_keys"] == ["another_extra", "weird_top"]


def test_empty_trajectory_yields_zero_events_and_no_crash():
    norm = normalize_row({"trajectory": []})
    assert norm["events"] == []
    assert norm["trajectory_length"] == 0


def test_missing_trajectory_treated_as_empty():
    norm = normalize_row({})  # no "trajectory" key
    assert norm["trajectory_length"] == 0


def test_non_dict_input_raises_type_error():
    with pytest.raises(TypeError):
        normalize_row(["not", "a", "dict"])


def test_non_list_trajectory_raises_type_error():
    with pytest.raises(TypeError):
        normalize_row({"trajectory": "oops"})


# ---------- invariants ----------


def test_trajectory_length_matches_events_length():
    row = {"trajectory": [_system("sp"), _user("issue"), _ai("thought\n```\nls\n```"), _user("a\nb")]}
    norm = normalize_row(row)
    assert norm["trajectory_length"] == len(norm["events"]) == 4


def test_step_index_is_dense_and_zero_based():
    row = {"trajectory": [_system("sp"), _user("issue"), _ai("t\n```\nls\n```")]}
    norm = normalize_row(row)
    assert [ev["step_index"] for ev in norm["events"]] == [0, 1, 2]


def test_schema_version_and_source_are_set():
    norm = normalize_row({"trajectory": []}, source="swe_agent_nebius")
    assert norm["schema_version"] == SCHEMA_VERSION
    assert norm["source"] == "swe_agent_nebius"


def test_raw_metadata_lengths_are_char_counts_of_strings_not_truthiness():
    norm = normalize_row({"trajectory": [], "generated_patch": "abc", "eval_logs": "xyz12"})
    assert norm["raw_metadata"]["patch_length"] == 3
    assert norm["raw_metadata"]["eval_logs_length"] == 5
    norm2 = normalize_row({"trajectory": [], "generated_patch": None, "eval_logs": None})
    assert norm2["raw_metadata"]["patch_length"] == 0
    assert norm2["raw_metadata"]["eval_logs_length"] == 0


# ---------- end-to-end on the real sample row ----------


SAMPLE_ROW = (
    Path(__file__).resolve().parents[1]
    / "external_data"
    / "swe_agent"
    / "raw"
    / "sample_row.json"
)


@pytest.mark.skipif(not SAMPLE_ROW.exists(), reason="sample_row.json not present")
def test_full_sample_row_round_trip_holds_invariants():
    row = json.loads(SAMPLE_ROW.read_text(encoding="utf-8"))
    norm = normalize_row(row)
    assert norm["instance_id"] == "AnalogJ__lexicon-336"
    assert norm["model_name"] == "swe-agent-llama-70b"
    assert norm["final_success"] is False
    assert norm["trajectory_length"] == 93 == len(norm["events"])
    assert norm["events"][0]["role"] == "system"
    assert norm["events"][1]["role"] == "environment"
    assert "ISSUE" in (norm["issue_text"] or "")
    assert norm["raw_metadata"]["patch_length"] == 2190
    assert norm["raw_metadata"]["eval_logs_length"] == 2048
    # Every assistant event has either a command or a parse_warning.
    for ev in norm["events"]:
        if ev["role"] == "assistant":
            assert ev["command"] is not None or ev["raw"].get("parse_warnings")


def test_main_writes_normalized_trace_and_summary(tmp_path):
    src = tmp_path / "row.json"
    src.write_text(json.dumps({
        "instance_id": "x__y-1",
        "model_name": "m",
        "target": True,
        "trajectory": [
            _system("sp"),
            _user("issue"),
            _ai("thought\n```\nls\n```"),
        ],
        "exit_status": "submitted (exit_context)",
        "generated_patch": "diff",
        "eval_logs": "ok",
    }), encoding="utf-8")
    run_dir = tmp_path / "run"
    rc = main(["--source-row", str(src), "--run-dir", str(run_dir)])
    assert rc == 0
    nt = json.loads((run_dir / "normalized_trace.json").read_text(encoding="utf-8"))
    assert nt["instance_id"] == "x__y-1"
    assert nt["trajectory_length"] == 3
    summary = (run_dir / "trajectory_summary.md").read_text(encoding="utf-8")
    assert "x__y-1" in summary
    assert "ls" in summary  # the assistant command is mentioned


def test_render_summary_truncates_with_explicit_marker():
    events = [_system("sp")] + [_user(f"obs-{i}") for i in range(80)]
    norm = normalize_row({"trajectory": events})
    out = render_summary(norm, max_steps=10)
    assert "more steps elided" in out
