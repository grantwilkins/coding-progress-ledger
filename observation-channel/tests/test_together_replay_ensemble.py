"""
Claim:
The Together replay runner asks each observer to estimate progress from the
original task prompt plus the observed work prefix available at that turn.

Plausible wrong implementations:
- Put the original task only in a separate system summary or omit it.
- Send only the current turn instead of all previous work seen so far.
- Leak future turns into an earlier observer prompt.
- Record agent estimates under the wrong turn after replaying a prefix.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    Estimate,
    aggregate,
    original_task_prompt,
    parse_fraction,
    replay_prompt,
    run_parallel_replay,
    system_prompt,
    turn_prompt,
    validate_replay_args,
)


def test_replay_prompt_pairs_original_task_with_observed_work_prefix() -> None:
    prompt = replay_prompt(
        "ISSUE:\nFix parser.py",
        ["Turn 1: USER.\nInitial report", "Turn 2: ACTION via open.\nopen parser.py"],
    )

    assert "Original task prompt:\nISSUE:\nFix parser.py" in prompt
    assert "Observed work so far:" in prompt
    assert "Initial report" in prompt
    assert "open parser.py" in prompt
    assert "final outcome" not in prompt.lower()


def test_system_prompt_contains_replay_rules_not_task_context() -> None:
    prompt = system_prompt()

    assert "Original task prompt:" not in prompt
    assert "observed work received so far" in prompt
    assert "No words, no JSON." in prompt


def test_turn_prompt_requires_original_task_prompt() -> None:
    prompt = turn_prompt(
        "ISSUE",
        Turn(step=3, kind="action", tool="search_dir", command='search_dir "needle"'),
    )

    assert "Original task prompt:\nISSUE" in prompt
    assert "Turn 3: ACTION via search_dir" in prompt
    assert 'search_dir "needle"' in prompt


def test_run_parallel_replay_sends_original_task_and_only_seen_work_prefixes() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append(messages)
        return str(len(calls) / 10)

    estimates = run_parallel_replay(
        [
            Turn(step=1, kind="user", response="bug report"),
            Turn(step=2, kind="action", tool="open", command="open parser.py"),
        ],
        agents=1,
        workers=1,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE:\nFix parser.py",
        complete=complete,
    )

    user_prompts = [
        next(message["content"] for message in call if message["role"] == "user")
        for call in calls
    ]
    assert len(user_prompts) == 2
    assert all("ISSUE:\nFix parser.py" in prompt for prompt in user_prompts)
    assert "bug report" in user_prompts[0]
    assert "open parser.py" not in user_prompts[0]
    assert "bug report" in user_prompts[1]
    assert "open parser.py" in user_prompts[1]
    assert [(estimate.turn, estimate.agent, estimate.value) for estimate in estimates] == [
        (1, 1, 0.1),
        (2, 1, 0.2),
    ]


def test_aggregate_groups_agent_estimates_by_turn() -> None:
    rows = aggregate(
        [
            Estimate(turn=1, agent=1, value=0.0, raw="0.0"),
            Estimate(turn=1, agent=2, value=0.2, raw="0.2"),
            Estimate(turn=2, agent=1, value=0.7, raw="0.7"),
            Estimate(turn=2, agent=2, value=0.9, raw="0.9"),
        ]
    )

    assert [row["turn"] for row in rows] == [1, 2]
    assert [row["mean_fraction_complete"] for row in rows] == [
        pytest.approx(0.1),
        pytest.approx(0.8),
    ]


def test_original_task_prompt_uses_first_raw_user_message() -> None:
    row = {
        "instance_id": "case",
        "trajectory": [
            {"role": "system", "text": "solver instructions"},
            {"role": "user", "text": "ISSUE:\nFix the parser"},
            {"role": "ai", "text": "searching"},
            {"role": "user", "text": "Found parser.py"},
        ],
    }

    assert original_task_prompt(row) == "ISSUE:\nFix the parser"


def test_fraction_and_replay_arg_validation_fail_loudly() -> None:
    assert parse_fraction("0.72") == pytest.approx(0.72)

    with pytest.raises(ValueError, match="no numeric fraction"):
        parse_fraction("complete enough")
    with pytest.raises(ValueError, match="out of range"):
        parse_fraction("1.2")
    with pytest.raises(ValueError, match="task_prompt is empty"):
        replay_prompt("", ["Turn 1: USER.\nIssue"])
    with pytest.raises(ValueError, match="agents must be positive"):
        validate_replay_args("m", 0, 1)
