"""
Claim:
Source readers convert visible trace actions/responses into ordered canonical
turns without using thoughts as evidence.

Plausible wrong implementations:
- Treat assistant narration as a command instead of extracting the fenced or structured command.
- Leave generic Hermes shell tools as `terminal`, hiding the actual command class.
- Treat mini-SWE user return tags as user text instead of observations.
- Silently accept malformed Hermes tool-call JSON.
"""

import pytest

from observation_channel.readers import hermes_turns, mini_swe_turns, swe_agent_turns
from observation_channel.readers import rows_to_turns


def test_swe_agent_reader_omits_thoughts_and_extracts_tool() -> None:
    row = {
        "instance_id": "repo__issue-1",
        "exit_status": "submitted",
        "trajectory": [
            {"role": "system", "system_prompt": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<think>hidden</think>\n```bash\ncreate foo.py\n```"},
            {"role": "user", "observation": "created"},
        ],
    }

    turns = swe_agent_turns(row)

    assert [turn.kind for turn in turns] == ["system", "user", "action", "observation"]
    assert turns[2].tool == "create"
    assert "hidden" not in (turns[2].command or "")


def test_mini_swe_reader_extracts_bash_and_return_tags() -> None:
    row = {
        "trial_id": "trial",
        "messages": [
            {"role": "assistant", "content": "```bash\npytest tests\n```"},
            {"role": "user", "content": "<returncode>1</returncode><output>failed</output>"},
        ],
    }

    turns = mini_swe_turns(row)

    assert turns[0].kind == "action"
    assert turns[0].command == "pytest tests"
    assert turns[1].kind == "observation"
    assert "returncode=1" in (turns[1].response or "")
    assert "failed" in (turns[1].response or "")


def test_hermes_reader_extracts_tool_call_and_response() -> None:
    row = {
        "id": "trace",
        "conversations": [
            {"from": "human", "value": "fix it"},
            {
                "from": "gpt",
                "value": "<think>hidden</think><tool_call>{\"name\":\"terminal\",\"arguments\":{\"command\":\"pytest\"}}</tool_call>",
            },
            {"from": "tool", "value": "<tool_response>{\"name\":\"terminal\",\"content\":\"failed\"}</tool_response>"},
        ],
    }

    turns = hermes_turns(row)

    assert [turn.kind for turn in turns] == ["user", "action", "observation"]
    assert turns[1].tool == "pytest"
    assert turns[1].command == "pytest"
    assert turns[1].arguments == {"command": "pytest"}
    assert turns[2].response == "failed"


def test_hermes_reader_fails_on_malformed_tool_call_json() -> None:
    row = {"id": "trace", "conversations": [{"from": "gpt", "value": "<tool_call>{bad}</tool_call>"}]}

    with pytest.raises(ValueError, match="tool_call contains invalid JSON"):
        hermes_turns(row)


def test_rows_to_turns_uses_later_nonempty_identifier() -> None:
    row = {
        "trial_id": "",
        "task_name": "task-a",
        "steps": [{"src": "agent", "tools": [{"cmd": "pytest"}], "msg": "", "obs": "<returncode>0</returncode>"}],
    }

    [(instance_id, turns)] = list(rows_to_turns([row], source="terminalbench"))

    assert instance_id == "task-a"
    assert turns[0].command == "pytest"


def test_mini_swe_reader_drops_placeholder_tool_commands() -> None:
    row = {
        "task_name": "task-a",
        "steps": [
            {"src": "agent", "tools": [{"cmd": "$38"}], "msg": "$37", "obs": "<returncode>0</returncode>"},
            {"src": "agent", "tools": [{"cmd": "pytest"}], "msg": "```bash\npytest\n```", "obs": ""},
        ],
    }

    [(instance_id, turns)] = list(rows_to_turns([row], source="terminalbench"))

    assert instance_id == "task-a"
    assert [turn.kind for turn in turns] == ["observation", "action"]
    assert turns[1].command == "pytest"
