"""
Claim:
The Together replay ensemble asks independent agents for one scalar progress
estimate per visible turn and ablation condition, dispatches agents in parallel
within each turn, derives smaller ensemble-size views by agent subsetting, and
aggregates progress values without leaking future turns or final labels.

Plausible wrong implementations:
- Include future turns, hidden evaluator results, or final trace status in the
  prompt for an early turn.
- Replace the original SWE-Agent issue prompt with a manual task summary.
- Extract the system prompt or a later observation instead of the first user task
  prompt from the raw trajectory.
- Parse arbitrary prose or out-of-range values as valid fractions.
- Reject unambiguous whole-number percentages like `90`, causing long live runs
  to fail after the model answered on a percent scale.
- Aggregate across the wrong level, such as across all turns instead of per turn.
- Advance an agent history without recording the scalar response for that turn.
- Treat remaining-work scores as progress instead of inverting them.
- Rerun model calls for every requested ensemble size instead of subsetting.
- Abort a long live run after one observer returns a blank or invalid response,
  instead of dropping only that failed sample.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    Estimate,
    agreement_matrix,
    aggregate,
    format_turn_distribution,
    original_task_prompt,
    parse_fraction,
    run_ablation_grid,
    run_parallel_replay,
    system_prompt,
    turn_evidence,
    turn_prompt,
    validate_ablation_args,
)


def estimate(
    *,
    turn: int,
    agent: int,
    prompt_variant: str = "fraction_complete",
    progress_value: float,
    raw_value: float | None = None,
) -> Estimate:
    return Estimate(
        trace_key="trace",
        turn=turn,
        agent=agent,
        model="model",
        prompt_variant=prompt_variant,
        context_variant="task_and_trace",
        ensemble_size=2,
        raw_value=progress_value if raw_value is None else raw_value,
        progress_value=progress_value,
        raw=str(progress_value if raw_value is None else raw_value),
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


def test_context_variants_hide_task_and_skip_observations_for_commands_only() -> None:
    assert "Original task prompt:" in system_prompt("ISSUE:\nsecret", "fraction_complete", "task_and_trace")
    assert "secret" not in system_prompt("ISSUE:\nsecret", "fraction_complete", "trace_only")
    assert turn_evidence(Turn(step=1, kind="observation", response="output"), "commands_only") is None
    assert "edit file.py" in (turn_evidence(Turn(step=2, kind="action", command="edit file.py"), "commands_only") or "")


def test_parse_fraction_accepts_single_fraction_and_rejects_bad_values() -> None:
    assert parse_fraction("0.72") == 0.72
    assert parse_fraction("90") == pytest.approx(0.9)
    assert parse_fraction("100%") == pytest.approx(1.0)

    with pytest.raises(ValueError):
        parse_fraction("complete enough")
    with pytest.raises(ValueError):
        parse_fraction("1.2")
    with pytest.raises(ValueError):
        parse_fraction("90.5")


def test_aggregate_is_per_turn_not_global() -> None:
    rows = aggregate(
        [
            estimate(turn=1, agent=1, progress_value=0.0),
            estimate(turn=1, agent=2, progress_value=0.2),
            estimate(turn=2, agent=1, progress_value=0.8),
            estimate(turn=2, agent=2, progress_value=1.0),
        ]
    )

    assert [row["turn"] for row in rows] == [1, 2]
    assert rows[0]["mean_progress"] == pytest.approx(0.1)
    assert rows[0]["stdev_progress"] == pytest.approx(0.1)
    assert rows[0]["median"] == pytest.approx(0.1)
    assert rows[1]["mean_progress"] == pytest.approx(0.9)
    assert rows[1]["stdev_progress"] == pytest.approx(0.1)


def test_grouped_aggregation_separates_prompt_variants() -> None:
    rows = aggregate(
        [
            estimate(turn=1, agent=1, prompt_variant="fraction_complete", progress_value=0.2),
            estimate(turn=1, agent=2, prompt_variant="fraction_complete", progress_value=0.4),
            estimate(turn=1, agent=1, prompt_variant="goal_closeness", progress_value=0.8),
            estimate(turn=1, agent=2, prompt_variant="goal_closeness", progress_value=1.0),
        ]
    )

    means = {row["prompt_variant"]: row["mean_progress"] for row in rows}
    assert means == {"fraction_complete": pytest.approx(0.3), "goal_closeness": pytest.approx(0.9)}


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
        trace_key="trace",
        agents=2,
        workers=2,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE:\nFix the test case",
        prompt_variant="fraction_complete",
        context_variant="task_and_trace",
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
        trace_key="trace",
        agents=2,
        workers=2,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE:\nOriginal SWE issue",
        prompt_variant="fraction_complete",
        context_variant="task_and_trace",
        progress=False,
        complete=complete,
    )

    assert len(system_messages) == 2
    assert all("Original SWE issue" in message for message in system_messages)
    assert all("High-level task:" not in message for message in system_messages)


def test_prompt_variant_conversion_inverts_remaining_work() -> None:
    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return "0.25"

    fraction = run_parallel_replay(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        agents=1,
        workers=1,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        prompt_variant="fraction_complete",
        context_variant="task_and_trace",
        complete=complete,
    )[0]
    remaining = run_parallel_replay(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        agents=1,
        workers=1,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        prompt_variant="remaining_work",
        context_variant="task_and_trace",
        complete=complete,
    )[0]

    assert fraction.raw_value == pytest.approx(0.25)
    assert fraction.progress_value == pytest.approx(0.25)
    assert remaining.raw_value == pytest.approx(0.25)
    assert remaining.progress_value == pytest.approx(0.75)


def test_parallel_replay_skips_single_failed_agent_but_fails_if_all_fail() -> None:
    responses = iter(["", "0.5"])

    def partly_failing_complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return next(responses)

    estimates = run_parallel_replay(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        agents=2,
        workers=1,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        prompt_variant="fraction_complete",
        context_variant="task_and_trace",
        complete=partly_failing_complete,
    )

    assert len(estimates) == 1
    assert estimates[0].progress_value == pytest.approx(0.5)

    def failing_complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return ""

    with pytest.raises(RuntimeError, match="all 2 agents failed"):
        run_parallel_replay(
            [Turn(step=1, kind="action", command="edit")],
            trace_key="trace",
            agents=2,
            workers=1,
            model="m",
            api_key="k",
            max_retries=0,
            max_tokens=8,
            temperature=0.0,
            task_prompt="ISSUE",
            prompt_variant="fraction_complete",
            context_variant="task_and_trace",
            complete=failing_complete,
        )


def test_progress_print_uses_successful_agents_after_partial_failure(capsys: pytest.CaptureFixture[str]) -> None:
    responses = iter(["0.25", "", "0.75"])

    def partly_failing_complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return next(responses)

    estimates = run_parallel_replay(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        agents=3,
        workers=1,
        model="m",
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        task_prompt="ISSUE",
        prompt_variant="fraction_complete",
        context_variant="task_and_trace",
        progress=True,
        complete=partly_failing_complete,
    )

    output = capsys.readouterr().out
    assert len(estimates) == 2
    assert "agent=2 failed" in output
    assert "turn=1 n=2" in output


def test_ablation_grid_subsets_ensemble_sizes_without_extra_calls() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append(messages)
        return "0.5"

    estimates = run_ablation_grid(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        task_prompt="ISSUE",
        models=["m"],
        prompt_variants=["fraction_complete"],
        context_variants=["task_and_trace"],
        ensemble_sizes=[1, 5, 10],
        workers=10,
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        progress=False,
        agents=10,
        complete=complete,
    )

    assert len(calls) == 10
    assert {estimate.ensemble_size for estimate in estimates} == {1, 5, 10}
    assert sum(1 for estimate in estimates if estimate.ensemble_size == 1) == 1
    assert sum(1 for estimate in estimates if estimate.ensemble_size == 5) == 5
    assert sum(1 for estimate in estimates if estimate.ensemble_size == 10) == 10


def test_agreement_matrix_identical_and_different_curves() -> None:
    identical = agreement_matrix(
        [
            {"model": "a", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 1, "mean_progress": 0.1},
            {"model": "a", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 2, "mean_progress": 0.9},
            {"model": "b", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 1, "mean_progress": 0.1},
            {"model": "b", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 2, "mean_progress": 0.9},
        ]
    )
    pair = next(row for row in identical if row["condition_a"].startswith("a |") and row["condition_b"].startswith("b |"))
    assert pair["correlation"] == pytest.approx(1.0)
    assert pair["mean_abs_difference"] == pytest.approx(0.0)

    different = agreement_matrix(
        [
            {"model": "a", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 1, "mean_progress": 0.1},
            {"model": "a", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 2, "mean_progress": 0.9},
            {"model": "b", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 1, "mean_progress": 0.9},
            {"model": "b", "prompt_variant": "p", "context_variant": "c", "ensemble_size": 1, "turn": 2, "mean_progress": 0.1},
        ]
    )
    pair = next(row for row in different if row["condition_a"].startswith("a |") and row["condition_b"].startswith("b |"))
    assert pair["correlation"] == pytest.approx(-1.0)
    assert pair["mean_abs_difference"] > 0


def test_invalid_ablation_args_fail_before_api_calls() -> None:
    with pytest.raises(ValueError):
        validate_ablation_args(["m"], ["bad"], ["task_and_trace"], [1], None)
    with pytest.raises(ValueError):
        validate_ablation_args(["m"], ["fraction_complete"], ["bad"], [1], None)
    with pytest.raises(ValueError):
        validate_ablation_args(["m"], ["fraction_complete"], ["task_and_trace"], [2], 1)


def test_turn_distribution_print_includes_mean_quantiles_and_histogram() -> None:
    text = format_turn_distribution(3, [0.0, 0.2, 0.8, 1.0])

    assert "turn=3" in text
    assert "mean=0.500" in text
    assert "median=0.500" in text
    assert "0.0-0.1:1" in text
    assert "0.8-0.9:1" in text
