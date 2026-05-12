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
DEFAULT_MODELS = (
    "openai/gpt-oss-20b,"
    "openai/gpt-oss-120b,"
    "Qwen/Qwen3-Coder-480B-A35B-Instruct"
)
DEFAULT_PROMPT_VARIANTS = "fraction_complete,remaining_work,goal_closeness"
DEFAULT_CONTEXTS = "task_and_trace,trace_only"
DEFAULT_ENSEMBLE_SIZES = "1,5,10,40"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "together_replay_ablation"
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
class Estimate:
    trace_key: str
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
    parser = argparse.ArgumentParser(description="Run a Together replay progress ablation on one SWE-Agent trace.")
    parser.add_argument("--raw-index", type=int, default=349)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--model", help="backward-compatible alias for --models with one model")
    parser.add_argument("--prompt-variants", default=DEFAULT_PROMPT_VARIANTS)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--ensemble-sizes", default=DEFAULT_ENSEMBLE_SIZES)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/hf_cache"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true", help="suppress per-turn aggregate progress prints")
    args = parser.parse_args(argv)

    models = parse_csv_arg(args.model or args.models)
    prompt_variants = parse_csv_arg(args.prompt_variants)
    context_variants = parse_csv_arg(args.contexts)
    ensemble_sizes = parse_int_csv_arg(args.ensemble_sizes)
    validate_ablation_args(models, prompt_variants, context_variants, ensemble_sizes, args.agents)
    max_agents = args.agents or max(ensemble_sizes)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    row = swe_agent_row(args.raw_index, args.cache_dir)
    [(instance_id, turns)] = list(rows_to_turns([row], source="swe-agent"))
    task_prompt = original_task_prompt(row)
    estimates = run_ablation_grid(
        turns,
        trace_key=f"swe-agent:{args.raw_index:06d}:{instance_id}",
        task_prompt=task_prompt,
        models=models,
        prompt_variants=prompt_variants,
        context_variants=context_variants,
        ensemble_sizes=ensemble_sizes,
        workers=args.workers,
        api_key=api_key,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        progress=not args.quiet,
        agents=max_agents,
    )
    write_report(args.report_dir, estimates, row, args, models, prompt_variants, context_variants, ensemble_sizes, max_agents)
    return 0


def swe_agent_row(raw_index: int, cache_dir: Path) -> dict[str, Any]:
    for index, row in enumerate(
        iter_hf_rows(source="swe-agent", cache_dir=cache_dir, split="train", limit=raw_index + 1, local_files_only=True)
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
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("CSV argument must contain at least one value")
    return values


def parse_int_csv_arg(value: str) -> list[int]:
    values = [int(part) for part in parse_csv_arg(value)]
    if any(value <= 0 for value in values):
        raise ValueError("ensemble sizes must be positive integers")
    return values


def validate_ablation_args(
    models: list[str],
    prompt_variants: list[str],
    context_variants: list[str],
    ensemble_sizes: list[int],
    agents: int | None,
) -> None:
    if not models:
        raise ValueError("at least one model is required")
    unknown_prompts = sorted(set(prompt_variants) - set(PROMPT_VARIANTS))
    if unknown_prompts:
        raise ValueError(f"unknown prompt variants: {unknown_prompts}")
    unknown_contexts = sorted(set(context_variants) - KNOWN_CONTEXTS)
    if unknown_contexts:
        raise ValueError(f"unknown context variants: {unknown_contexts}")
    if not ensemble_sizes:
        raise ValueError("at least one ensemble size is required")
    if agents is not None and max(ensemble_sizes) > agents:
        raise ValueError("max ensemble size cannot exceed --agents")


def run_parallel_replay(
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
) -> list[Estimate]:
    complete = complete or together_complete
    transcript: list[str] = []
    estimates: list[Estimate] = []
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
            estimates.append(
                Estimate(
                    trace_key=trace_key,
                    turn=turn.step,
                    agent=agent,
                    model=model,
                    prompt_variant=prompt_variant,
                    context_variant=context_variant,
                    ensemble_size=agents,
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
    return estimates


def to_progress_value(raw_value: float, prompt_variant: str) -> float:
    if PROMPT_VARIANTS[prompt_variant]["parse_as"] == "remaining":
        return 1.0 - raw_value
    return raw_value


def run_ablation_grid(
    turns: list[Turn],
    *,
    trace_key: str,
    task_prompt: str,
    models: list[str],
    prompt_variants: list[str],
    context_variants: list[str],
    ensemble_sizes: list[int],
    workers: int,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    progress: bool,
    agents: int,
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str] | None = None,
) -> list[Estimate]:
    estimates: list[Estimate] = []
    for model in models:
        for prompt_variant in prompt_variants:
            for context_variant in context_variants:
                sampled = run_parallel_replay(
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
                for size in ensemble_sizes:
                    estimates.extend(
                        Estimate(
                            trace_key=estimate.trace_key,
                            turn=estimate.turn,
                            agent=estimate.agent,
                            model=estimate.model,
                            prompt_variant=estimate.prompt_variant,
                            context_variant=estimate.context_variant,
                            ensemble_size=size,
                            raw_value=estimate.raw_value,
                            progress_value=estimate.progress_value,
                            raw=estimate.raw,
                        )
                        for estimate in sampled
                        if estimate.agent <= size
                    )
    return estimates


def system_prompt(task_prompt: str, prompt_variant: str = "fraction_complete", context_variant: str = "task_and_trace") -> str:
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
    evidence = turn_evidence(turn, "task_and_trace")
    if evidence is None:
        raise ValueError("turn produced no evidence")
    return replay_prompt([evidence], "fraction_complete")


def replay_prompt(turns_so_far: list[str], prompt_variant: str = "fraction_complete") -> str:
    return (
        "Observed turns so far:\n\n"
        + "\n\n".join(turns_so_far)
        + "\n\n"
        + str(PROMPT_VARIANTS[prompt_variant]["user_question"])
    )


def turn_evidence(turn: Turn, context_variant: str = "task_and_trace") -> str | None:
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


def aggregate(estimates: Iterable[Estimate]) -> list[dict[str, Any]]:
    by_turn: dict[tuple[str, str, str, str, int, int], list[float]] = {}
    for estimate in estimates:
        key = (
            estimate.trace_key,
            estimate.model,
            estimate.prompt_variant,
            estimate.context_variant,
            estimate.ensemble_size,
            estimate.turn,
        )
        by_turn.setdefault(key, []).append(estimate.progress_value)
    rows = []
    for (trace_key, model, prompt_variant, context_variant, ensemble_size, turn), values in sorted(by_turn.items()):
        ordered = sorted(values)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stdev = math.sqrt(variance)
        rows.append(
            {
                "trace_key": trace_key,
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
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in aggregate_rows:
        key = (
            str(row["trace_key"]),
            str(row["model"]),
            str(row["prompt_variant"]),
            str(row["context_variant"]),
            int(row["ensemble_size"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (trace_key, model, prompt_variant, context_variant, ensemble_size), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["turn"]))
        means = [float(row["mean_progress"]) for row in ordered]
        deltas = [means[index] - means[index - 1] for index in range(1, len(means))]
        summaries.append(
            {
                "trace_key": trace_key,
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
        label = condition_label(row)
        curves.setdefault(label, {})[int(row["turn"])] = float(row["mean_progress"])
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


def condition_label(row: dict[str, Any]) -> str:
    return f"{row['model']} | {row['prompt_variant']} | {row['context_variant']} | n={row['ensemble_size']}"


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
    histogram = " ".join(f"{i/10:.1f}-{(i+1)/10:.1f}:{count}" for i, count in enumerate(bins) if count)
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


def together_complete(messages: list[dict[str, str]], model: str, api_key: str, _max_retries: int, max_tokens: int, temperature: float) -> str:
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
    estimates: list[Estimate],
    row: dict[str, Any],
    args: argparse.Namespace,
    models: list[str],
    prompt_variants: list[str],
    context_variants: list[str],
    ensemble_sizes: list[int],
    max_agents: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_estimates(report_dir / "agent_estimates.csv", estimates)
    aggregate_rows = aggregate(estimates)
    _write_rows(report_dir / "aggregate_progress.csv", aggregate_rows)
    summary_rows = condition_summary(aggregate_rows)
    _write_rows(report_dir / "condition_summary.csv", summary_rows)
    agreement_rows = agreement_matrix(aggregate_rows)
    _write_rows(report_dir / "agreement_matrix.csv", agreement_rows)
    main_model = models[0]
    main_prompt = "fraction_complete" if "fraction_complete" in prompt_variants else prompt_variants[0]
    main_context = "task_and_trace" if "task_and_trace" in context_variants else context_variants[0]
    main_size = max(ensemble_sizes)
    _plot(report_dir / "aggregate_progress.png", aggregate_rows, main_model, main_prompt, main_context, main_size)
    _plot_trajectories(report_dir / "agent_trajectories.png", estimates, aggregate_rows, main_model, main_prompt, main_context, main_size)
    _plot_comparison(
        report_dir / "prompt_comparison.png",
        aggregate_rows,
        fixed={"model": main_model, "context_variant": main_context, "ensemble_size": main_size},
        vary="prompt_variant",
        title="Prompt wording comparison",
    )
    _plot_comparison(
        report_dir / "model_comparison.png",
        aggregate_rows,
        fixed={"prompt_variant": main_prompt, "context_variant": main_context, "ensemble_size": main_size},
        vary="model",
        title="Observer model comparison",
    )
    _plot_comparison(
        report_dir / "context_comparison.png",
        aggregate_rows,
        fixed={"model": main_model, "prompt_variant": main_prompt, "ensemble_size": main_size},
        vary="context_variant",
        title="Context comparison",
    )
    _plot_ensemble_size_convergence(
        report_dir / "ensemble_size_convergence.png", aggregate_rows, main_model, main_prompt, main_context, ensemble_sizes
    )
    _plot_agreement_matrix(report_dir / "agreement_matrix.png", agreement_rows)
    _write_readme(report_dir, row, args, summary_rows, models, prompt_variants, context_variants, ensemble_sizes, max_agents)


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


def _write_estimates(path: Path, estimates: list[Estimate]) -> None:
    _write_rows(path, (estimate.__dict__ for estimate in estimates))


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    prompt_variant: str,
    context_variant: str,
    ensemble_size: int,
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if row["model"] == model
            and row["prompt_variant"] == prompt_variant
            and row["context_variant"] == context_variant
            and int(row["ensemble_size"]) == ensemble_size
        ],
        key=lambda row: int(row["turn"]),
    )


def _plot(
    path: Path,
    rows: list[dict[str, Any]],
    model: str,
    prompt_variant: str,
    context_variant: str,
    ensemble_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _select_rows(rows, model=model, prompt_variant=prompt_variant, context_variant=context_variant, ensemble_size=ensemble_size)
    if not rows:
        raise ValueError("no rows for aggregate progress plot")
    x = [int(row["turn"]) for row in rows]
    y = [float(row["mean_progress"]) for row in rows]
    lo = [float(row["lower_1sd"]) for row in rows]
    hi = [float(row["upper_1sd"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.fill_between(x, lo, hi, alpha=0.2, color="#59A14F", linewidth=0)
    axis.plot(x, y, color="#2E6B2E", linewidth=2)
    axis.scatter(x, y, s=16, color="#2E6B2E")
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("mean progress")
    axis.set_title("Together replay ablation main condition")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trajectories(
    path: Path,
    estimates: list[Estimate],
    aggregate_rows: list[dict[str, Any]],
    model: str,
    prompt_variant: str,
    context_variant: str,
    ensemble_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_agent: dict[int, list[Estimate]] = {}
    for estimate in estimates:
        if (
            estimate.model == model
            and estimate.prompt_variant == prompt_variant
            and estimate.context_variant == context_variant
            and estimate.ensemble_size == ensemble_size
        ):
            by_agent.setdefault(estimate.agent, []).append(estimate)
    aggregate_rows = _select_rows(
        aggregate_rows, model=model, prompt_variant=prompt_variant, context_variant=context_variant, ensemble_size=ensemble_size
    )
    if not by_agent or not aggregate_rows:
        raise ValueError("no rows for agent trajectory plot")
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
        [int(row["turn"]) for row in aggregate_rows],
        [float(row["mean_progress"]) for row in aggregate_rows],
        color="#2E6B2E",
        linewidth=2.2,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("fraction complete")
    axis.set_title("Together replay agent trajectories")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_comparison(path: Path, rows: list[dict[str, Any]], fixed: dict[str, object], vary: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if all(row[key] == value for key, value in fixed.items())]
    if not selected:
        raise ValueError(f"no rows for {title}")
    fig, axis = plt.subplots(figsize=(8, 3.2))
    for value in sorted({row[vary] for row in selected}):
        line_rows = sorted([row for row in selected if row[vary] == value], key=lambda row: int(row["turn"]))
        axis.plot([int(row["turn"]) for row in line_rows], [float(row["mean_progress"]) for row in line_rows], linewidth=1.8, label=str(value))
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("mean progress")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_ensemble_size_convergence(
    path: Path,
    rows: list[dict[str, Any]],
    model: str,
    prompt_variant: str,
    context_variant: str,
    ensemble_sizes: list[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["model"] == model and row["prompt_variant"] == prompt_variant and row["context_variant"] == context_variant
    ]
    if not selected:
        raise ValueError("no rows for ensemble convergence plot")
    max_size = max(ensemble_sizes)
    max_curve = {
        int(row["turn"]): float(row["mean_progress"]) for row in selected if int(row["ensemble_size"]) == max_size
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    deviations = []
    for size in sorted(ensemble_sizes):
        line_rows = sorted([row for row in selected if int(row["ensemble_size"]) == size], key=lambda row: int(row["turn"]))
        axes[0].plot([int(row["turn"]) for row in line_rows], [float(row["mean_progress"]) for row in line_rows], label=f"n={size}")
        diffs = [abs(float(row["mean_progress"]) - max_curve[int(row["turn"])]) for row in line_rows if int(row["turn"]) in max_curve]
        deviations.append((size, sum(diffs) / len(diffs) if diffs else 0.0))
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("turn")
    axes[0].set_ylabel("mean progress")
    axes[0].set_title("Curves by ensemble size")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].plot([size for size, _ in deviations], [value for _, value in deviations], marker="o", color="#2E6B2E")
    axes[1].set_xlabel("ensemble size")
    axes[1].set_ylabel("mean abs deviation")
    axes[1].set_title(f"Deviation from n={max_size}")
    axes[1].grid(True, alpha=0.25)
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
    size = max(5.0, min(14.0, len(labels) * 0.22))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=5)
    axis.set_yticklabels(labels, fontsize=5)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Turn-curve agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_readme(
    report_dir: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    models: list[str],
    prompt_variants: list[str],
    context_variants: list[str],
    ensemble_sizes: list[int],
    max_agents: int,
) -> None:
    final_lines = [
        f"- `{condition_label(summary)}`: {float(summary['final_mean_progress']):.3f}"
        for summary in sorted(summary_rows, key=condition_label)
    ]
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Together Replay Progress Ablation",
                "",
                "Observer-derived progress measurements for one SWE-Agent trace.",
                "",
                f"- raw row index: `{args.raw_index}`",
                f"- instance: `{row.get('instance_id', '')}`",
                f"- selected models: `{', '.join(models)}`",
                f"- selected prompt variants: `{', '.join(prompt_variants)}`",
                f"- selected contexts: `{', '.join(context_variants)}`",
                f"- selected ensemble sizes: `{', '.join(str(size) for size in ensemble_sizes)}`",
                f"- max agents queried per condition: {max_agents}",
                f"- parallel workers per turn: {args.workers}",
                f"- temperature: {args.temperature}",
                f"- max tokens: {args.max_tokens}",
                f"- dataset target label: `{row.get('target', '')}`",
                "",
                "Smaller ensemble sizes are derived by prefix-subsetting agent IDs from the max-agent run.",
                "",
                "These are observer-derived progress measurements, not ground-truth progress labels.",
                "The target label and evaluator logs are not included in per-turn prompts.",
                "",
                "Final progress by condition:",
                "",
                *final_lines,
                "",
                "Artifacts:",
                "",
                "- `agent_estimates.csv`: one scalar estimate per agent per turn.",
                "- `aggregate_progress.csv`: mean, standard deviation, and one-sigma band by turn.",
                "- `condition_summary.csv`: stability metrics by condition.",
                "- `agreement_matrix.csv`: pairwise curve agreement by condition.",
                "- `aggregate_progress.png`: main-condition aggregate progress curve.",
                "- `agent_trajectories.png`: main-condition agent trajectories with the mean overlaid.",
                "- `prompt_comparison.png`: prompt wording comparison.",
                "- `model_comparison.png`: observer model comparison.",
                "- `context_comparison.png`: context comparison.",
                "- `ensemble_size_convergence.png`: ensemble-size curves and deviation from the max ensemble.",
                "- `agreement_matrix.png`: pairwise correlation heatmap.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
