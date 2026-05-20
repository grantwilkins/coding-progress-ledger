"""
Claim:
The Together replay runner asks one observer to estimate seconds remaining and
confidence from the original task prompt plus the observed work prefix available
at that turn.

Plausible wrong implementations:
- Accept free-form or malformed model output that cannot be plotted as seconds.
- Send only the current turn instead of all previous work seen so far.
- Leak future turns into an earlier observer prompt.
- Record a retried invalid response instead of the first valid estimate.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    Estimate,
    original_task_prompt,
    parse_time_estimate,
    replay_prompt,
    run_replay,
    system_prompt,
    turn_prompt,
    validate_replay_args,
)


def test_replay_prompt_pairs_original_task_with_observed_work_prefix() -> None:
    prompt = replay_prompt(
        "ISSUE:\nFix parser.py",
        ["Turn 1: USER.\nInitial report", "Turn 2: ACTION via open.\nopen parser.py"],
    )

    assert "Original prompt:\nISSUE:\nFix parser.py" in prompt
    assert "State up until this point:" in prompt
    assert "Initial report" in prompt
    assert "open parser.py" in prompt
    assert "{XXs, YY%}" in prompt


def test_system_prompt_contains_time_contract_not_task_context() -> None:
    prompt = system_prompt()

    assert "Original prompt:" not in prompt
    assert "observed turns" in prompt
    assert "Return only {XXs, YY%}" in prompt


def test_turn_prompt_requires_original_task_prompt() -> None:
    prompt = turn_prompt(
        "ISSUE",
        Turn(step=3, kind="action", tool="search_dir", command='search_dir "needle"'),
    )

    assert "Original prompt:\nISSUE" in prompt
    assert "Turn 3: ACTION via search_dir" in prompt
    assert 'search_dir "needle"' in prompt


def test_run_replay_sends_original_task_and_only_seen_work_prefixes() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append(messages)
        return f"{{{len(calls) * 10}s, 70%}}"

    estimates = run_replay(
        [
            Turn(step=1, kind="user", response="bug report"),
            Turn(step=2, kind="action", tool="open", command="open parser.py"),
        ],
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
    assert estimates == [
        Estimate(turn=1, seconds_left=10.0, confidence_percent=70.0, raw="{10s, 70%}"),
        Estimate(turn=2, seconds_left=20.0, confidence_percent=70.0, raw="{20s, 70%}"),
    ]


def test_run_replay_retries_nonsense_before_recording_estimate(monkeypatch) -> None:
    monkeypatch.setattr(
        "observation_channel.together_replay_ensemble.time.sleep", lambda _: None
    )
    responses = iter(["probably soon", "{90s, 60%}"])

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return next(responses)

    estimates = run_replay(
        [Turn(step=1, kind="user", response="bug report")],
        model="m",
        api_key="k",
        max_retries=1,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        complete=complete,
    )

    assert estimates == [
        Estimate(turn=1, seconds_left=90.0, confidence_percent=60.0, raw="{90s, 60%}")
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


def test_time_estimate_and_replay_arg_validation_fail_loudly() -> None:
    assert parse_time_estimate("{120s, 75%}") == pytest.approx((120.0, 75.0))
    assert parse_time_estimate("{12.5s, 66.5%}") == pytest.approx((12.5, 66.5))

    with pytest.raises(ValueError, match="invalid time estimate"):
        parse_time_estimate("120 seconds, 75 percent")
    with pytest.raises(ValueError, match="confidence out of range"):
        parse_time_estimate("{120s, 101%}")
    with pytest.raises(ValueError, match="task_prompt is empty"):
        replay_prompt("", ["Turn 1: USER.\nIssue"])
    with pytest.raises(ValueError, match="model is required"):
        validate_replay_args("")
