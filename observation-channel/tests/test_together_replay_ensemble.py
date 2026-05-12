"""
Claim:
The Together replay ensemble asks independent agents for one scalar progress
estimate per visible turn, dispatches agents in parallel within each turn, and
aggregates their estimates into a mean and uncertainty band without leaking
future turns or final labels.

Plausible wrong implementations:
- Include future turns, hidden evaluator results, or final trace status in the
  prompt for an early turn.
- Replace the original SWE-Agent issue prompt with a manual task summary.
- Extract the system prompt or a later observation instead of the first user task
  prompt from the raw trajectory.
- Parse arbitrary prose or out-of-range values as valid fractions.
- Aggregate across the wrong level, such as across all turns instead of per turn.
- Advance an agent history without recording the scalar response for that turn.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    Estimate,
    aggregate,
    format_turn_distribution,
    original_task_prompt,
    parse_fraction,
    run_parallel_replay,
    system_prompt,
    turn_prompt,
)


def test_turn_prompt_contains_only_current_turn_evidence() -> None:
    prompt = turn_prompt(Turn(step=3, kind="action", tool="search_dir", command='search_dir "Hyphen not allowed"'))

    assert "Turn 3" in prompt
    assert 'search_dir "Hyphen not allowed"' in prompt
    assert "future" not in prompt.lower()
    assert "target=True" not in prompt
    assert "passed" not in prompt.lower()


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


def test_system_prompt_contains_original_task_prompt_not_manual_summary() -> None:
    prompt = system_prompt("ISSUE:\nAllow x-y identifiers")

    assert "Original task prompt:" in prompt
    assert "Allow x-y identifiers" in prompt
    assert "High-level task:" not in prompt
    assert "biomedsheets" not in prompt


def test_parse_fraction_accepts_single_fraction_and_rejects_bad_values() -> None:
    assert parse_fraction("0.72") == 0.72

    with pytest.raises(ValueError):
        parse_fraction("complete enough")
    with pytest.raises(ValueError):
        parse_fraction("1.2")


def test_aggregate_is_per_turn_not_global() -> None:
    rows = aggregate(
        [
            Estimate(turn=1, agent=1, value=0.0, raw="0"),
            Estimate(turn=1, agent=2, value=0.2, raw="0.2"),
            Estimate(turn=2, agent=1, value=0.8, raw="0.8"),
            Estimate(turn=2, agent=2, value=1.0, raw="1"),
        ]
    )

    assert [row["turn"] for row in rows] == [1, 2]
    assert rows[0]["mean_fraction_complete"] == pytest.approx(0.1)
    assert rows[0]["stdev"] == pytest.approx(0.1)
    assert rows[1]["mean_fraction_complete"] == pytest.approx(0.9)
    assert rows[1]["stdev"] == pytest.approx(0.1)


def test_parallel_replay_replays_only_turns_seen_so_far_and_records_each_turn() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append([message["content"] for message in messages if message["role"] == "user"])
        return str(len(calls) / 10)

    estimates = run_parallel_replay(
        [
            Turn(step=1, kind="user", response="issue text"),
            Turn(step=2, kind="action", tool="open", command="open file.py"),
        ],
        agents=2,
        workers=2,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE:\nFix the test case",
        progress=False,
        complete=complete,
    )

    assert [(estimate.turn, estimate.agent) for estimate in estimates] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(len(call) == 1 for call in calls[:2])
    assert all(len(call) == 1 for call in calls[2:])
    assert all("open file.py" not in call[0] for call in calls[:2])
    assert all("issue text" in call[0] and "open file.py" in call[0] for call in calls[2:])


def test_parallel_replay_sends_original_task_prompt_to_every_agent_call() -> None:
    system_messages = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        system_messages.append(messages[0]["content"])
        return "0.5"

    run_parallel_replay(
        [Turn(step=1, kind="action", tool="edit", command="edit file.py")],
        agents=2,
        workers=2,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE:\nOriginal SWE issue",
        progress=False,
        complete=complete,
    )

    assert len(system_messages) == 2
    assert all("Original SWE issue" in message for message in system_messages)
    assert all("High-level task:" not in message for message in system_messages)


def test_turn_distribution_print_includes_mean_quantiles_and_histogram() -> None:
    text = format_turn_distribution(3, [0.0, 0.2, 0.8, 1.0])

    assert "turn=3" in text
    assert "mean=0.500" in text
    assert "median=0.500" in text
    assert "0.0-0.1:1" in text
    assert "0.8-0.9:1" in text
