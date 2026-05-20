"""
Claim:
The Together replay runner asks each configured observer model to estimate
seconds remaining and confidence from the original task prompt plus the observed
work prefix available at that turn.

Plausible wrong implementations:
- Accept free-form or malformed model output that cannot be plotted as seconds.
- Scale error bars by the estimate instead of using inverse confidence directly.
- Mix estimates from different models under the wrong legend label.
- Send only the current turn instead of all previous work seen so far.
- Leak future turns into an earlier observer prompt.
- Record a retried invalid response instead of the first valid estimate.
- Fail to document that the selected SWE-Agent trace resolves successfully.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    Estimate,
    original_task_prompt,
    parse_time_estimate,
    replay_prompt,
    run_model_replays,
    run_replay,
    swe_agent_success_evidence,
    system_prompt,
    time_error_seconds,
    _streamed_content,
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
        Estimate("m", 1, 10.0, 70.0, "{10s, 70%}"),
        Estimate("m", 2, 20.0, 70.0, "{20s, 70%}"),
    ]


def test_run_model_replays_preserves_model_labels() -> None:
    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return "{10s, 70%}" if model == "small" else "{20s, 80%}"

    estimates = run_model_replays(
        [Turn(step=1, kind="user", response="bug report")],
        models=["small", "large"],
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        complete=complete,
    )

    assert estimates == [
        Estimate("small", 1, 10.0, 70.0, "{10s, 70%}"),
        Estimate("large", 1, 20.0, 80.0, "{20s, 80%}"),
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
        Estimate("m", 1, 90.0, 60.0, "{90s, 60%}")
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
        validate_replay_args([""])
    with pytest.raises(ValueError, match="at least one model"):
        validate_replay_args([])


def test_streamed_content_collects_delta_content_only() -> None:
    lines = [
        b'data: {"choices":[{"delta":{"reasoning":"hidden","content":"{12"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"s, 80%"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"}"}}]}\n',
        b"data: [DONE]\n",
    ]

    assert _streamed_content(lines) == "{12s, 80%}"


def test_time_error_seconds_uses_inverse_confidence_points() -> None:
    assert time_error_seconds(Estimate("m", 1, 120.0, 75.0, "{120s, 75%}")) == pytest.approx(
        25.0
    )
    assert time_error_seconds(Estimate("m", 1, 120.0, 100.0, "{120s, 100%}")) == pytest.approx(
        0.0
    )


def test_swe_agent_success_evidence_requires_target_and_passed_eval_logs() -> None:
    row = {
        "target": True,
        "exit_status": "submitted",
        "eval_logs": "tests/test_io_tsv_germline.py ....... [100%]\n7 passed",
    }

    assert swe_agent_success_evidence(row) == {
        "target": True,
        "exit_status": "submitted",
        "eval_passed": True,
    }
