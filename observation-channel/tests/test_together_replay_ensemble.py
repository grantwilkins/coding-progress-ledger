"""
Claim:
The Together replay runner measures one fixed observer baseline and explicit
single-axis deviations, reusing raw observer samples for smaller ensemble-size
conditions and aggregating reported progress by named condition and turn.

Plausible wrong implementations:
- Recreate a prompt x context x ensemble Cartesian grid instead of named
  directional conditions.
- Rerun API calls for every ensemble size rather than prefix-subsetting agents.
- Treat remaining-work estimates as progress instead of inverting them.
- Leak the task prompt into trace-only prompts or observations into commands-only
  transcripts.
- Aggregate raw samples by protocol fields while losing the condition boundary.
- Accept unknown prompt or context names after live API calls have started.
"""

import pytest

from observation_channel.models import Turn
from observation_channel.together_replay_ensemble import (
    ConditionEstimate,
    DEFAULT_AGENTS,
    DEFAULT_BASELINE_CONTEXT,
    DEFAULT_BASELINE_PROMPT,
    DEFAULT_CONTEXT_ABLATIONS,
    DEFAULT_ENSEMBLE_ABLATIONS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_ABLATIONS,
    aggregate,
    agreement_matrix,
    build_directional_conditions,
    format_turn_distribution,
    original_task_prompt,
    parse_csv_arg,
    parse_fraction,
    run_directional_replay,
    run_observer_samples,
    system_prompt,
    turn_evidence,
    turn_prompt,
    validate_directional_args,
)


def condition_estimate(
    *,
    condition: str,
    turn: int,
    agent: int,
    progress_value: float,
    ablation_axis: str = "baseline",
    prompt_variant: str = "fraction_complete",
    ensemble_size: int = 2,
) -> ConditionEstimate:
    return ConditionEstimate(
        trace_key="trace",
        condition=condition,
        ablation_axis=ablation_axis,
        turn=turn,
        agent=agent,
        model="model",
        prompt_variant=prompt_variant,
        context_variant="task_and_trace",
        ensemble_size=ensemble_size,
        raw_value=progress_value,
        progress_value=progress_value,
        raw=str(progress_value),
    )


def test_default_config_is_fixed_baseline_protocol() -> None:
    assert DEFAULT_MODEL == "openai/gpt-oss-120b"
    assert DEFAULT_BASELINE_PROMPT == "fraction_complete"
    assert DEFAULT_BASELINE_CONTEXT == "task_and_trace"
    assert DEFAULT_AGENTS == 40
    assert DEFAULT_PROMPT_ABLATIONS == "remaining_work,goal_closeness"
    assert DEFAULT_ENSEMBLE_ABLATIONS == "1,5,10,20"
    assert DEFAULT_CONTEXT_ABLATIONS == ""


def test_build_directional_conditions_returns_only_single_axis_deviations() -> None:
    conditions = build_directional_conditions(
        model="m",
        baseline_prompt="fraction_complete",
        baseline_context="task_and_trace",
        agents=40,
        prompt_ablations=["remaining_work", "goal_closeness"],
        ensemble_ablations=[1, 5, 10, 20],
        context_ablations=["trace_only"],
    )

    assert [condition.name for condition in conditions] == [
        "baseline",
        "prompt_remaining_work",
        "prompt_goal_closeness",
        "ensemble_n_1",
        "ensemble_n_5",
        "ensemble_n_10",
        "ensemble_n_20",
        "context_trace_only",
    ]
    assert all(condition.model == "m" for condition in conditions)
    assert {
        (condition.ablation_axis, condition.prompt_variant, condition.context_variant, condition.ensemble_size)
        for condition in conditions
    } == {
        ("baseline", "fraction_complete", "task_and_trace", 40),
        ("prompt", "remaining_work", "task_and_trace", 40),
        ("prompt", "goal_closeness", "task_and_trace", 40),
        ("ensemble", "fraction_complete", "task_and_trace", 1),
        ("ensemble", "fraction_complete", "task_and_trace", 5),
        ("ensemble", "fraction_complete", "task_and_trace", 10),
        ("ensemble", "fraction_complete", "task_and_trace", 20),
        ("context", "fraction_complete", "trace_only", 40),
    }


def test_ensemble_ablations_reuse_baseline_samples_by_agent_subset() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append(messages)
        return "0.5"

    conditions = build_directional_conditions(
        model="m",
        baseline_prompt="fraction_complete",
        baseline_context="task_and_trace",
        agents=5,
        prompt_ablations=[],
        ensemble_ablations=[1, 3],
        context_ablations=[],
    )
    samples, estimates = run_directional_replay(
        [Turn(step=1, kind="action", command="edit")],
        trace_key="trace",
        task_prompt="ISSUE",
        conditions=conditions,
        agents=5,
        workers=1,
        api_key="k",
        max_retries=0,
        max_tokens=8,
        temperature=0.0,
        progress=False,
        complete=complete,
    )

    assert len(calls) == 5
    assert len(samples) == 5
    counts = {}
    for estimate in estimates:
        counts[estimate.condition] = counts.get(estimate.condition, 0) + 1
    assert counts == {"baseline": 5, "ensemble_n_1": 1, "ensemble_n_3": 3}


def test_remaining_work_converts_to_progress_value() -> None:
    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return "0.25"

    sample = run_observer_samples(
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

    assert sample.raw_value == pytest.approx(0.25)
    assert sample.progress_value == pytest.approx(0.75)


def test_trace_only_hides_task_prompt_and_task_and_trace_includes_it() -> None:
    assert "Original task prompt:" in system_prompt("ISSUE:\nsecret", "fraction_complete", "task_and_trace")
    assert "secret" in system_prompt("ISSUE:\nsecret", "fraction_complete", "task_and_trace")
    assert "Original task prompt:" not in system_prompt("ISSUE:\nsecret", "fraction_complete", "trace_only")
    assert "secret" not in system_prompt("ISSUE:\nsecret", "fraction_complete", "trace_only")


def test_commands_only_skips_observation_evidence() -> None:
    assert turn_evidence(Turn(step=1, kind="observation", response="output"), "commands_only") is None
    assert "edit file.py" in (turn_evidence(Turn(step=2, kind="action", command="edit file.py"), "commands_only") or "")


def test_aggregate_groups_by_condition_and_turn() -> None:
    rows = aggregate(
        [
            condition_estimate(condition="baseline", turn=1, agent=1, progress_value=0.0),
            condition_estimate(condition="baseline", turn=1, agent=2, progress_value=0.2),
            condition_estimate(condition="prompt_remaining_work", turn=1, agent=1, progress_value=0.8, ablation_axis="prompt"),
            condition_estimate(condition="prompt_remaining_work", turn=1, agent=2, progress_value=1.0, ablation_axis="prompt"),
        ]
    )

    means = {row["condition"]: row["mean_progress"] for row in rows}
    assert means == {"baseline": pytest.approx(0.1), "prompt_remaining_work": pytest.approx(0.9)}
    assert all(row["turn"] == 1 for row in rows)


def test_unknown_prompt_or_context_fails_before_api_calls() -> None:
    with pytest.raises(ValueError, match="unknown prompt variants"):
        validate_directional_args("m", ["bad"], [], [1], 40)
    with pytest.raises(ValueError, match="unknown context variants"):
        validate_directional_args("m", [], ["bad"], [1], 40)
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_directional_args("m", [], [], [41], 40)


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


def test_parse_fraction_accepts_fraction_and_whole_percent_only() -> None:
    assert parse_fraction("0.72") == 0.72
    assert parse_fraction("90") == pytest.approx(0.9)
    assert parse_fraction("100%") == pytest.approx(1.0)

    with pytest.raises(ValueError):
        parse_fraction("complete enough")
    with pytest.raises(ValueError):
        parse_fraction("1.2")
    with pytest.raises(ValueError):
        parse_fraction("90.5")


def test_observer_replay_replays_only_turns_seen_so_far_and_records_each_turn() -> None:
    calls = []

    def complete(messages, model, api_key, max_retries, max_tokens, temperature):
        calls.append([message["content"] for message in messages if message["role"] == "user"])
        return str(len(calls) / 10)

    samples = run_observer_samples(
        [
            Turn(step=1, kind="user", response="issue text"),
            Turn(step=2, kind="action", tool="open", command="open file.py"),
        ],
        trace_key="trace",
        agents=2,
        workers=1,
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

    assert [(sample.turn, sample.agent) for sample in samples] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all("open file.py" not in call[0] for call in calls[:2])
    assert all("issue text" in call[0] and "open file.py" in call[0] for call in calls[2:])


def test_observer_replay_skips_single_failed_agent_but_fails_if_all_fail() -> None:
    responses = iter(["", "0.5"])

    def partly_failing_complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return next(responses)

    samples = run_observer_samples(
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

    assert len(samples) == 1
    assert samples[0].progress_value == pytest.approx(0.5)

    def failing_complete(messages, model, api_key, max_retries, max_tokens, temperature):
        return ""

    with pytest.raises(RuntimeError, match="all 2 agents failed"):
        run_observer_samples(
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

    samples = run_observer_samples(
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
    assert len(samples) == 2
    assert "agent=2 failed" in output
    assert "turn=1 n=2" in output


def test_agreement_matrix_compares_condition_curves() -> None:
    rows = [
        {"condition": "baseline", "turn": 1, "mean_progress": 0.1},
        {"condition": "baseline", "turn": 2, "mean_progress": 0.9},
        {"condition": "ensemble_n_1", "turn": 1, "mean_progress": 0.1},
        {"condition": "ensemble_n_1", "turn": 2, "mean_progress": 0.9},
        {"condition": "prompt_remaining_work", "turn": 1, "mean_progress": 0.9},
        {"condition": "prompt_remaining_work", "turn": 2, "mean_progress": 0.1},
    ]

    matrix = agreement_matrix(rows)
    same = next(row for row in matrix if row["condition_a"] == "baseline" and row["condition_b"] == "ensemble_n_1")
    opposite = next(row for row in matrix if row["condition_a"] == "baseline" and row["condition_b"] == "prompt_remaining_work")
    assert same["correlation"] == pytest.approx(1.0)
    assert same["mean_abs_difference"] == pytest.approx(0.0)
    assert opposite["correlation"] == pytest.approx(-1.0)
    assert opposite["mean_abs_difference"] > 0


def test_parse_csv_arg_allows_empty_context_ablation_default() -> None:
    assert parse_csv_arg("") == []


def test_turn_distribution_print_includes_mean_quantiles_and_histogram() -> None:
    text = format_turn_distribution(3, [0.0, 0.2, 0.8, 1.0])

    assert "turn=3" in text
    assert "mean=0.500" in text
    assert "median=0.500" in text
    assert "0.0-0.1:1" in text
    assert "0.8-0.9:1" in text
