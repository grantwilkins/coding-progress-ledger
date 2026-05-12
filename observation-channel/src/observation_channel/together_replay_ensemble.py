from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .hf import iter_hf_rows
from .models import Turn
from .readers import rows_to_turns


TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASELINE_PROMPT = "fraction_complete"
DEFAULT_BASELINE_CONTEXT = "task_and_trace"
DEFAULT_AGENTS = 40
DEFAULT_PROMPT_ABLATIONS = "remaining_work,goal_closeness"
DEFAULT_ENSEMBLE_ABLATIONS = "1,5,10,20"
DEFAULT_CONTEXT_ABLATIONS = ""
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "together_replay_directional"
KNOWN_CONTEXTS = {"task_and_trace", "trace_only", "commands_only"}
PROMPT_VARIANTS = {
    "fraction_complete": {
        "system_suffix": (
            "After each turn, reply with only one number from 0 to 1: "
            "the fraction of the total work you estimate is complete. "
            "No words, no JSON."
        ),
        "user_question": "On a scale of 0 to 1, what fraction of the work do you estimate is complete?",
        "parse_as": "progress",
    },
    "remaining_work": {
        "system_suffix": (
            "After each turn, reply with only one number from 0 to 1: "
            "the fraction of the total work you estimate remains. "
            "No words, no JSON."
        ),
        "user_question": "On a scale of 0 to 1, what fraction of the work do you estimate remains?",
        "parse_as": "remaining",
    },
    "goal_closeness": {
        "system_suffix": (
            "After each turn, reply with only one number from 0 to 1: "
            "how close the agent appears to be to solving the task, where 0 is no useful progress "
            "and 1 is solved or essentially solved. "
            "No words, no JSON."
        ),
        "user_question": "On a scale of 0 to 1, how close is the agent to solving the task?",
        "parse_as": "progress",
    },
    "success_likelihood": {
        "system_suffix": (
            "After each turn, reply with only one number from 0 to 1: "
            "the probability that the current run will solve the task without major rework. "
            "No words, no JSON."
        ),
        "user_question": "On a scale of 0 to 1, how likely is the current run to solve the task without major rework?",
        "parse_as": "progress",
    },
}


@dataclass(frozen=True)
class Condition:
    name: str
    model: str
    prompt_variant: str
    context_variant: str
    ensemble_size: int
    ablation_axis: str


@dataclass(frozen=True)
class ObserverSample:
    trace_key: str
    turn: int
    agent: int
    model: str
    prompt_variant: str
    context_variant: str
    raw_value: float
    progress_value: float
    raw: str


@dataclass(frozen=True)
class ConditionEstimate:
    trace_key: str
    condition: str
    ablation_axis: str
    turn: int
    agent: int
    model: str
    prompt_variant: str
    context_variant: str
    ensemble_size: int
    raw_value: float
    progress_value: float
    raw: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run directional Together replay progress ablations on one SWE-Agent trace."
    )
    parser.add_argument("--raw-index", type=int, default=349)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--prompt-ablations", default=DEFAULT_PROMPT_ABLATIONS)
    parser.add_argument("--ensemble-ablations", default=DEFAULT_ENSEMBLE_ABLATIONS)
    parser.add_argument("--context-ablations", default=DEFAULT_CONTEXT_ABLATIONS)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/hf_cache"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true", help="suppress per-turn aggregate progress prints")
    args = parser.parse_args(argv)

    prompt_ablations = parse_csv_arg(args.prompt_ablations)
    ensemble_ablations = parse_int_csv_arg(args.ensemble_ablations)
    context_ablations = parse_csv_arg(args.context_ablations)
    validate_directional_args(args.model, prompt_ablations, context_ablations, ensemble_ablations, args.agents)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    row = swe_agent_row(args.raw_index, args.cache_dir)
    [(instance_id, turns)] = list(rows_to_turns([row], source="swe-agent"))
    task_prompt = original_task_prompt(row)
    trace_key = f"swe-agent:{args.raw_index:06d}:{instance_id}"
    conditions = build_directional_conditions(
        model=args.model,
        baseline_prompt=DEFAULT_BASELINE_PROMPT,
        baseline_context=DEFAULT_BASELINE_CONTEXT,
        agents=args.agents,
        prompt_ablations=prompt_ablations,
        ensemble_ablations=ensemble_ablations,
        context_ablations=context_ablations,
    )
    samples, estimates = run_directional_replay(
        turns,
        trace_key=trace_key,
        task_prompt=task_prompt,
        conditions=conditions,
        agents=args.agents,
        workers=args.workers,
        api_key=api_key,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        progress=not args.quiet,
    )
    write_report(args.report_dir, samples, estimates, row, args, conditions)
    return 0


def swe_agent_row(raw_index: int, cache_dir: Path) -> dict[str, Any]:
    for index, row in enumerate(
        iter_hf_rows(
            source="swe-agent",
            cache_dir=cache_dir,
            split="train",
            limit=raw_index + 1,
            local_files_only=True,
        )
    ):
        if index == raw_index:
            return row
    raise ValueError(f"raw index {raw_index} not found")


def original_task_prompt(row: dict[str, Any]) -> str:
    for item in row.get("trajectory") or []:
        if str(item.get("role", "")).lower() in {"user", "human"}:
            text = str(item.get("text") or item.get("content") or item.get("message") or "").strip()
            if text:
                return text
    raise ValueError(f"{row.get('instance_id', '<unknown>')}: no original user task prompt found")


def parse_csv_arg(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv_arg(value: str) -> list[int]:
    values = [int(part) for part in parse_csv_arg(value)]
    if any(value <= 0 for value in values):
        raise ValueError("ensemble sizes must be positive integers")
    return values


def validate_directional_args(
    model: str,
    prompt_ablations: list[str],
    context_ablations: list[str],
    ensemble_ablations: list[int],
    agents: int,
) -> None:
    if not model.strip():
        raise ValueError("model is required")
    if agents <= 0:
        raise ValueError("agents must be positive")
    if not ensemble_ablations:
        raise ValueError("at least one ensemble ablation is required")
    unknown_prompts = sorted(set(prompt_ablations + [DEFAULT_BASELINE_PROMPT]) - set(PROMPT_VARIANTS))
    if unknown_prompts:
        raise ValueError(f"unknown prompt variants: {unknown_prompts}")
    unknown_contexts = sorted(set(context_ablations + [DEFAULT_BASELINE_CONTEXT]) - KNOWN_CONTEXTS)
    if unknown_contexts:
        raise ValueError(f"unknown context variants: {unknown_contexts}")
    if any(size > agents for size in ensemble_ablations):
        raise ValueError("ensemble ablations cannot exceed --agents")


def build_directional_conditions(
    *,
    model: str,
    baseline_prompt: str,
    baseline_context: str,
    agents: int,
    prompt_ablations: list[str],
    ensemble_ablations: list[int],
    context_ablations: list[str],
) -> list[Condition]:
    if baseline_prompt not in PROMPT_VARIANTS:
        raise ValueError(f"unknown baseline prompt variant: {baseline_prompt}")
    if baseline_context not in KNOWN_CONTEXTS:
        raise ValueError(f"unknown baseline context variant: {baseline_context}")
    if any(size <= 0 for size in ensemble_ablations):
        raise ValueError("ensemble ablations must be positive")
    if any(size > agents for size in ensemble_ablations):
        raise ValueError("ensemble ablations cannot exceed agents")
    conditions = [
        Condition("baseline", model, baseline_prompt, baseline_context, agents, "baseline"),
    ]
    seen = {"baseline"}
    for prompt in prompt_ablations:
        if prompt == baseline_prompt:
            continue
        name = f"prompt_{prompt}"
        if name not in seen:
            conditions.append(Condition(name, model, prompt, baseline_context, agents, "prompt"))
            seen.add(name)
    for size in sorted(set(ensemble_ablations)):
        if size == agents:
            continue
        name = f"ensemble_n_{size}"
        if name not in seen:
            conditions.append(Condition(name, model, baseline_prompt, baseline_context, size, "ensemble"))
            seen.add(name)
    for context in context_ablations:
        if context == baseline_context:
            continue
        name = f"context_{context}"
        if name not in seen:
            conditions.append(Condition(name, model, baseline_prompt, context, agents, "context"))
            seen.add(name)
    return conditions


def run_directional_replay(
    turns: list[Turn],
    *,
    trace_key: str,
    task_prompt: str,
    conditions: list[Condition],
    agents: int,
    workers: int,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    progress: bool,
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str] | None = None,
) -> tuple[list[ObserverSample], list[ConditionEstimate]]:
    samples_by_protocol: dict[tuple[str, str, str], list[ObserverSample]] = {}
    for model, prompt_variant, context_variant in dict.fromkeys(
        (condition.model, condition.prompt_variant, condition.context_variant) for condition in conditions
    ):
        samples_by_protocol[(model, prompt_variant, context_variant)] = run_observer_samples(
            turns,
            trace_key=trace_key,
            agents=agents,
            workers=workers,
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            task_prompt=task_prompt,
            prompt_variant=prompt_variant,
            context_variant=context_variant,
            progress=progress,
            complete=complete,
        )
    samples = [sample for protocol_samples in samples_by_protocol.values() for sample in protocol_samples]
    estimates = derive_condition_estimates(conditions, samples_by_protocol)
    return samples, estimates


def run_observer_samples(
    turns: list[Turn],
    *,
    trace_key: str,
    agents: int,
    workers: int,
    model: str,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    task_prompt: str,
    prompt_variant: str,
    context_variant: str,
    progress: bool = False,
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str] | None = None,
) -> list[ObserverSample]:
    complete = complete or together_complete
    transcript: list[str] = []
    samples: list[ObserverSample] = []
    for turn in turns:
        evidence = turn_evidence(turn, context_variant)
        if evidence is not None:
            transcript.append(evidence)
        prompt = replay_prompt(transcript, prompt_variant)
        with ThreadPoolExecutor(max_workers=min(workers, agents)) as pool:
            futures = {
                pool.submit(
                    _complete_with_retries,
                    complete,
                    [
                        {"role": "system", "content": system_prompt(task_prompt, prompt_variant, context_variant)},
                        {"role": "user", "content": prompt},
                    ],
                    model,
                    api_key,
                    max_retries,
                    max_tokens,
                    temperature,
                ): agent
                for agent in range(1, agents + 1)
            }
            results = {}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    raw = future.result()
                except Exception as exc:
                    if progress:
                        print(f"{model} {prompt_variant} {context_variant} turn={turn.step} agent={agent} failed: {exc}", flush=True)
                    continue
                raw_value = parse_fraction(raw)
                progress_value = to_progress_value(raw_value, prompt_variant)
                results[agent] = (raw, raw_value, progress_value)
        if not results:
            raise RuntimeError(f"turn {turn.step} all {agents} agents failed")
        for agent in sorted(results):
            raw, raw_value, progress_value = results[agent]
            samples.append(
                ObserverSample(
                    trace_key=trace_key,
                    turn=turn.step,
                    agent=agent,
                    model=model,
                    prompt_variant=prompt_variant,
                    context_variant=context_variant,
                    raw_value=raw_value,
                    progress_value=progress_value,
                    raw=raw.strip(),
                )
            )
        if progress:
            print(
                f"{model} {prompt_variant} {context_variant} "
                + format_turn_distribution(turn.step, [results[agent][2] for agent in sorted(results)]),
                flush=True,
            )
    return samples


def derive_condition_estimates(
    conditions: list[Condition],
    samples_by_protocol: dict[tuple[str, str, str], list[ObserverSample]],
) -> list[ConditionEstimate]:
    estimates: list[ConditionEstimate] = []
    for condition in conditions:
        samples = samples_by_protocol[(condition.model, condition.prompt_variant, condition.context_variant)]
        estimates.extend(
            ConditionEstimate(
                trace_key=sample.trace_key,
                condition=condition.name,
                ablation_axis=condition.ablation_axis,
                turn=sample.turn,
                agent=sample.agent,
                model=sample.model,
                prompt_variant=sample.prompt_variant,
                context_variant=sample.context_variant,
                ensemble_size=condition.ensemble_size,
                raw_value=sample.raw_value,
                progress_value=sample.progress_value,
                raw=sample.raw,
            )
            for sample in samples
            if sample.agent <= condition.ensemble_size
        )
    return estimates


def to_progress_value(raw_value: float, prompt_variant: str) -> float:
    if PROMPT_VARIANTS[prompt_variant]["parse_as"] == "remaining":
        return 1.0 - raw_value
    return raw_value


def system_prompt(
    task_prompt: str,
    prompt_variant: str = DEFAULT_BASELINE_PROMPT,
    context_variant: str = DEFAULT_BASELINE_CONTEXT,
) -> str:
    suffix = PROMPT_VARIANTS[prompt_variant]["system_suffix"]
    prompt_parts = [
        "You are one member of a blind replay ensemble for a coding trace. "
        "You will receive one turn at a time. Use only the evidence received so far; "
        "do not assume future turns, final outcome, hidden tests, final trace length, or hidden labels."
    ]
    if context_variant == "task_and_trace":
        task_prompt = task_prompt.strip()
        if not task_prompt:
            raise ValueError("task_prompt is empty")
        prompt_parts.append(f"Original task prompt:\n{task_prompt}")
    prompt_parts.append(str(suffix))
    return "\n\n".join(prompt_parts)


def turn_prompt(turn: Turn) -> str:
    evidence = turn_evidence(turn, DEFAULT_BASELINE_CONTEXT)
    if evidence is None:
        raise ValueError("turn produced no evidence")
    return replay_prompt([evidence], DEFAULT_BASELINE_PROMPT)


def replay_prompt(turns_so_far: list[str], prompt_variant: str = DEFAULT_BASELINE_PROMPT) -> str:
    return (
        "Observed turns so far:\n\n"
        + "\n\n".join(turns_so_far)
        + "\n\n"
        + str(PROMPT_VARIANTS[prompt_variant]["user_question"])
    )


def turn_evidence(turn: Turn, context_variant: str = DEFAULT_BASELINE_CONTEXT) -> str | None:
    if context_variant == "commands_only" and turn.kind != "action":
        return None
    return f"Turn {turn.step}: {turn.kind.upper()}{' via ' + turn.tool if turn.tool else ''}.\n{_turn_text(turn)}"


def parse_fraction(text: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"no numeric fraction in response: {text!r}")
    value = float(match.group(0))
    if 0.0 <= value <= 1.0:
        return value
    if value.is_integer() and 1.0 < value <= 100.0:
        return value / 100.0
    raise ValueError(f"fraction out of range: {value}")


def aggregate(estimates: Iterable[ConditionEstimate]) -> list[dict[str, Any]]:
    by_condition: dict[tuple[str, str, str, str, str, str, int, int], list[float]] = {}
    for estimate in estimates:
        key = (
            estimate.trace_key,
            estimate.condition,
            estimate.ablation_axis,
            estimate.model,
            estimate.prompt_variant,
            estimate.context_variant,
            estimate.ensemble_size,
            estimate.turn,
        )
        by_condition.setdefault(key, []).append(estimate.progress_value)
    rows = []
    for (
        trace_key,
        condition,
        ablation_axis,
        model,
        prompt_variant,
        context_variant,
        ensemble_size,
        turn,
    ), values in sorted(by_condition.items()):
        ordered = sorted(values)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stdev = math.sqrt(variance)
        rows.append(
            {
                "trace_key": trace_key,
                "condition": condition,
                "ablation_axis": ablation_axis,
                "model": model,
                "prompt_variant": prompt_variant,
                "context_variant": context_variant,
                "ensemble_size": ensemble_size,
                "turn": turn,
                "n": len(values),
                "mean_progress": mean,
                "stdev_progress": stdev,
                "se_progress": stdev / math.sqrt(len(values)),
                "lower_1sd": max(0.0, mean - stdev),
                "upper_1sd": min(1.0, mean + stdev),
                "q25": _quantile(ordered, 0.25),
                "median": _quantile(ordered, 0.5),
                "q75": _quantile(ordered, 0.75),
                "min": ordered[0],
                "max": ordered[-1],
            }
        )
    return rows


def condition_summary(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        key = (
            str(row["trace_key"]),
            str(row["condition"]),
            str(row["ablation_axis"]),
            str(row["model"]),
            str(row["prompt_variant"]),
            str(row["context_variant"]),
            int(row["ensemble_size"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (trace_key, condition, ablation_axis, model, prompt_variant, context_variant, ensemble_size), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["turn"]))
        means = [float(row["mean_progress"]) for row in ordered]
        deltas = [means[index] - means[index - 1] for index in range(1, len(means))]
        summaries.append(
            {
                "trace_key": trace_key,
                "condition": condition,
                "ablation_axis": ablation_axis,
                "model": model,
                "prompt_variant": prompt_variant,
                "context_variant": context_variant,
                "ensemble_size": ensemble_size,
                "turns": len(ordered),
                "final_mean_progress": means[-1],
                "final_stdev_progress": float(ordered[-1]["stdev_progress"]),
                "mean_turn_stdev": sum(float(row["stdev_progress"]) for row in ordered) / len(ordered),
                "mean_abs_turn_delta": sum(abs(delta) for delta in deltas) / len(deltas) if deltas else 0.0,
                "large_jump_rate": sum(1 for delta in deltas if abs(delta) >= 0.20) / len(deltas) if deltas else 0.0,
                "monotonicity_violation_rate": sum(1 for delta in deltas if delta <= -0.10) / len(deltas) if deltas else 0.0,
                "area_under_progress_curve": sum(means) / len(means),
            }
        )
    return summaries


def agreement_matrix(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curves: dict[str, dict[int, float]] = {}
    for row in aggregate_rows:
        curves.setdefault(str(row["condition"]), {})[int(row["turn"])] = float(row["mean_progress"])
    rows = []
    labels = sorted(curves)
    for i, label_a in enumerate(labels):
        for label_b in labels[i:]:
            shared = sorted(set(curves[label_a]) & set(curves[label_b]))
            a = [curves[label_a][turn] for turn in shared]
            b = [curves[label_b][turn] for turn in shared]
            diffs = [abs(left - right) for left, right in zip(a, b)]
            rows.append(
                {
                    "condition_a": label_a,
                    "condition_b": label_b,
                    "correlation": correlation(a, b),
                    "mean_abs_difference": sum(diffs) / len(diffs) if diffs else 0.0,
                    "max_abs_difference": max(diffs) if diffs else 0.0,
                    "shared_turns": len(shared),
                }
            )
    return rows


def correlation(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("correlation inputs must have equal length")
    if len(a) < 2:
        return 1.0 if a == b else 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    numerator = sum((left - mean_a) * (right - mean_b) for left, right in zip(a, b))
    denom_a = math.sqrt(sum((left - mean_a) ** 2 for left in a))
    denom_b = math.sqrt(sum((right - mean_b) ** 2 for right in b))
    if denom_a == 0 or denom_b == 0:
        return 1.0 if a == b else 0.0
    return numerator / (denom_a * denom_b)


def format_turn_distribution(turn: int, values: list[float]) -> str:
    if not values:
        raise ValueError("cannot summarize empty values")
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    q1 = _quantile(ordered, 0.25)
    median = _quantile(ordered, 0.5)
    q3 = _quantile(ordered, 0.75)
    bins = [0] * 10
    for value in ordered:
        bins[min(9, int(value * 10))] += 1
    histogram = " ".join(f"{i / 10:.1f}-{(i + 1) / 10:.1f}:{count}" for i, count in enumerate(bins) if count)
    return (
        f"turn={turn} n={len(ordered)} mean={mean:.3f} sd={math.sqrt(variance):.3f} "
        f"min={ordered[0]:.3f} q1={q1:.3f} median={median:.3f} q3={q3:.3f} max={ordered[-1]:.3f} "
        f"hist={histogram}"
    )


def _quantile(ordered: list[float], p: float) -> float:
    if not ordered:
        raise ValueError("empty quantile")
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def together_complete(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    _max_retries: int,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        TOGETHER_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "observation-channel/0.1",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def write_report(
    report_dir: Path,
    samples: list[ObserverSample],
    estimates: list[ConditionEstimate],
    row: dict[str, Any],
    args: argparse.Namespace,
    conditions: list[Condition],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(report_dir / "observer_samples.csv", (sample.__dict__ for sample in samples))
    _write_rows(report_dir / "condition_estimates.csv", (estimate.__dict__ for estimate in estimates))
    aggregate_rows = aggregate(estimates)
    _write_rows(report_dir / "aggregate_progress.csv", aggregate_rows)
    summary_rows = condition_summary(aggregate_rows)
    _write_rows(report_dir / "condition_summary.csv", summary_rows)
    agreement_rows = agreement_matrix(aggregate_rows)
    _write_rows(report_dir / "agreement_matrix.csv", agreement_rows)
    _plot_condition(report_dir / "baseline_progress.png", aggregate_rows, "baseline", "Baseline progress")
    _plot_condition_trajectories(report_dir / "baseline_agent_trajectories.png", estimates, aggregate_rows, "baseline")
    _plot_axis(report_dir / "prompt_ablation.png", aggregate_rows, {"baseline", *{c.name for c in conditions if c.ablation_axis == "prompt"}}, "Prompt ablation")
    _plot_axis(report_dir / "ensemble_ablation.png", aggregate_rows, {"baseline", *{c.name for c in conditions if c.ablation_axis == "ensemble"}}, "Ensemble ablation")
    context_conditions = {c.name for c in conditions if c.ablation_axis == "context"}
    if context_conditions:
        _plot_axis(report_dir / "context_ablation.png", aggregate_rows, {"baseline", *context_conditions}, "Context ablation")
    else:
        _plot_placeholder(report_dir / "context_ablation.png", "No context ablations requested")
    _plot_agreement_matrix(report_dir / "agreement_matrix.png", agreement_rows)
    _write_readme(report_dir, row, args, summary_rows, conditions)


def _complete_with_retries(
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str],
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            raw = complete(messages, model, api_key, max_retries, max_tokens, temperature)
            parse_fraction(raw)
            return raw
        except ValueError:
            if attempt == max_retries:
                raise
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "Your previous response was invalid. Reply with only one numeric value from 0 to 1, like 0.42.",
                },
            ]
            time.sleep(2**attempt)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Together API HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _turn_text(turn: Turn) -> str:
    text = turn.command if turn.kind == "action" else turn.response
    return " ".join((text or "").split())[:2000]


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return sorted([row for row in rows if row["condition"] == condition], key=lambda row: int(row["turn"]))


def _plot_condition(path: Path, rows: list[dict[str, Any]], condition: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = _condition_rows(rows, condition)
    if not selected:
        raise ValueError(f"no rows for {condition}")
    fig, axis = plt.subplots(figsize=(8, 3.2))
    x = [int(row["turn"]) for row in selected]
    y = [float(row["mean_progress"]) for row in selected]
    lo = [float(row["lower_1sd"]) for row in selected]
    hi = [float(row["upper_1sd"]) for row in selected]
    axis.fill_between(x, lo, hi, alpha=0.2, color="#59A14F", linewidth=0)
    axis.plot(x, y, color="#2E6B2E", linewidth=2)
    axis.scatter(x, y, s=16, color="#2E6B2E")
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("mean progress")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_condition_trajectories(
    path: Path,
    estimates: list[ConditionEstimate],
    aggregate_rows: list[dict[str, Any]],
    condition: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_agent: dict[int, list[ConditionEstimate]] = {}
    for estimate in estimates:
        if estimate.condition == condition:
            by_agent.setdefault(estimate.agent, []).append(estimate)
    selected = _condition_rows(aggregate_rows, condition)
    if not by_agent or not selected:
        raise ValueError(f"no rows for {condition}")
    fig, axis = plt.subplots(figsize=(8, 3.2))
    for agent_estimates in by_agent.values():
        ordered = sorted(agent_estimates, key=lambda estimate: estimate.turn)
        axis.plot(
            [estimate.turn for estimate in ordered],
            [estimate.progress_value for estimate in ordered],
            color="#9B9B9B",
            alpha=0.22,
            linewidth=0.9,
        )
    axis.plot(
        [int(row["turn"]) for row in selected],
        [float(row["mean_progress"]) for row in selected],
        color="#2E6B2E",
        linewidth=2.2,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("fraction complete")
    axis.set_title("Baseline observer trajectories")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_axis(path: Path, rows: list[dict[str, Any]], conditions: set[str], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3.2))
    plotted = False
    for condition in sorted(conditions):
        selected = _condition_rows(rows, condition)
        if selected:
            plotted = True
            axis.plot(
                [int(row["turn"]) for row in selected],
                [float(row["mean_progress"]) for row in selected],
                linewidth=1.8,
                label=condition,
            )
    if not plotted:
        raise ValueError(f"no rows for {title}")
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("mean progress")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_placeholder(path: Path, text: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.text(0.5, 0.5, text, ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_agreement_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({row["condition_a"] for row in rows} | {row["condition_b"] for row in rows})
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0.0 for _ in labels] for _ in labels]
    for row in rows:
        i = index[str(row["condition_a"])]
        j = index[str(row["condition_b"])]
        matrix[i][j] = matrix[j][i] = float(row["correlation"])
    size = max(5.0, min(14.0, len(labels) * 0.45))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Condition agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_readme(
    report_dir: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    conditions: list[Condition],
) -> None:
    final_lines = [
        f"- `{summary['condition']}` ({summary['ablation_axis']}): {float(summary['final_mean_progress']):.3f}"
        for summary in sorted(summary_rows, key=lambda row: str(row["condition"]))
    ]
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Together Replay Directional Progress Ablation",
                "",
                "Observer-derived progress measurements for one SWE-Agent trace.",
                "",
                f"- raw row index: `{args.raw_index}`",
                f"- instance: `{row.get('instance_id', '')}`",
                f"- baseline model: `{args.model}`",
                f"- baseline prompt: `{DEFAULT_BASELINE_PROMPT}`",
                f"- baseline context: `{DEFAULT_BASELINE_CONTEXT}`",
                f"- baseline agents: {args.agents}",
                f"- prompt ablations: `{', '.join(condition.prompt_variant for condition in conditions if condition.ablation_axis == 'prompt')}`",
                f"- ensemble ablations: `{', '.join(str(condition.ensemble_size) for condition in conditions if condition.ablation_axis == 'ensemble')}`",
                f"- context ablations: `{', '.join(condition.context_variant for condition in conditions if condition.ablation_axis == 'context')}`",
                f"- parallel workers per turn: {args.workers}",
                f"- temperature: {args.temperature}",
                f"- max tokens: {args.max_tokens}",
                f"- dataset target label: `{row.get('target', '')}`",
                "",
                "This report is directional, not a Cartesian grid.",
                "The baseline is fixed.",
                "Every ablation changes exactly one axis from the baseline.",
                "Observer progress is not ground-truth progress.",
                "The target label and evaluator logs are not included in per-turn prompts.",
                f"Smaller ensemble sizes are derived from the same n={args.agents} observer run by prefix-subsetting agent IDs.",
                "",
                "Final progress by condition:",
                "",
                *final_lines,
                "",
                "Artifacts:",
                "",
                "- `observer_samples.csv`: one raw observer sample per actual API response.",
                "- `condition_estimates.csv`: observer samples projected into named directional conditions.",
                "- `aggregate_progress.csv`: mean, standard deviation, and one-sigma band by condition and turn.",
                "- `condition_summary.csv`: stability metrics by condition.",
                "- `agreement_matrix.csv`: pairwise curve agreement by condition.",
                "- `baseline_progress.png`: baseline aggregate progress curve.",
                "- `baseline_agent_trajectories.png`: baseline agent trajectories with the mean overlaid.",
                "- `prompt_ablation.png`: prompt-direction comparison.",
                "- `ensemble_ablation.png`: ensemble-size comparison.",
                "- `context_ablation.png`: context-direction comparison or placeholder.",
                "- `agreement_matrix.png`: pairwise correlation heatmap.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
