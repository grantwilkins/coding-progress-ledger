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
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "together_replay_ensemble"


@dataclass(frozen=True)
class Estimate:
    turn: int
    agent: int
    value: float
    raw: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a parallel Together replay ensemble on a SWE-Agent trace.")
    parser.add_argument("--raw-index", type=int, default=349)
    parser.add_argument("--agents", type=int, default=40)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/hf_cache"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true", help="suppress per-turn aggregate progress prints")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    row = swe_agent_row(args.raw_index, args.cache_dir)
    [(_, turns)] = list(rows_to_turns([row], source="swe-agent"))
    estimates = run_parallel_replay(
        turns,
        agents=args.agents,
        workers=args.workers,
        model=args.model,
        api_key=api_key,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        progress=not args.quiet,
    )
    write_report(args.report_dir, estimates, row, args)
    return 0


def swe_agent_row(raw_index: int, cache_dir: Path) -> dict[str, Any]:
    for index, row in enumerate(
        iter_hf_rows(source="swe-agent", cache_dir=cache_dir, split="train", limit=raw_index + 1, local_files_only=True)
    ):
        if index == raw_index:
            return row
    raise ValueError(f"raw index {raw_index} not found")


def run_parallel_replay(
    turns: list[Turn],
    *,
    agents: int,
    workers: int,
    model: str,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    progress: bool = False,
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str] | None = None,
) -> list[Estimate]:
    complete = complete or together_complete
    transcript: list[str] = []
    estimates: list[Estimate] = []
    for turn in turns:
        transcript.append(turn_evidence(turn))
        prompt = replay_prompt(transcript)
        with ThreadPoolExecutor(max_workers=min(workers, agents)) as pool:
            futures = {
                pool.submit(
                    _complete_with_retries,
                    complete,
                    [{"role": "system", "content": system_prompt()}, {"role": "user", "content": prompt}],
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
                    raise RuntimeError(f"turn {turn.step} agent {agent} failed") from exc
                value = parse_fraction(raw)
                results[agent] = (raw, value)
        for agent in range(1, agents + 1):
            raw, value = results[agent]
            estimates.append(Estimate(turn=turn.step, agent=agent, value=value, raw=raw.strip()))
        if progress:
            print(format_turn_distribution(turn.step, [results[agent][1] for agent in range(1, agents + 1)]), flush=True)
    return estimates


def system_prompt() -> str:
    return (
        "You are one member of a blind replay ensemble for a coding trace. "
        "You will receive one turn at a time. Use only the evidence received so far; "
        "do not assume future turns, final outcome, hidden tests, final trace length, or hidden labels.\n\n"
        "High-level task: In biomedsheets, fix the germline TSV reader so hyphens are allowed in "
        "SODAR sample sheet identifiers, especially patient/father/mother name columns where current validation rejects hyphens.\n\n"
        "After each turn, reply with only one number from 0 to 1: the fraction of work you estimate is complete. "
        "No words, no JSON."
    )


def turn_prompt(turn: Turn) -> str:
    return replay_prompt([turn_evidence(turn)])


def replay_prompt(turns_so_far: list[str]) -> str:
    return (
        "Observed turns so far:\n\n"
        + "\n\n".join(turns_so_far)
        + "\n\n"
        "On a scale of 0 to 1, what fraction of the work do you estimate is complete?"
    )


def turn_evidence(turn: Turn) -> str:
    return f"Turn {turn.step}: {turn.kind.upper()}{' via ' + turn.tool if turn.tool else ''}.\n{_turn_text(turn)}"


def parse_fraction(text: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"no numeric fraction in response: {text!r}")
    value = float(match.group(0))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"fraction out of range: {value}")
    return value


def aggregate(estimates: Iterable[Estimate]) -> list[dict[str, Any]]:
    by_turn: dict[int, list[float]] = {}
    for estimate in estimates:
        by_turn.setdefault(estimate.turn, []).append(estimate.value)
    rows = []
    for turn, values in sorted(by_turn.items()):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stdev = math.sqrt(variance)
        rows.append(
            {
                "turn": turn,
                "n": len(values),
                "mean_fraction_complete": mean,
                "stdev": stdev,
                "lower_1sd": max(0.0, mean - stdev),
                "upper_1sd": min(1.0, mean + stdev),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


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


def write_report(report_dir: Path, estimates: list[Estimate], row: dict[str, Any], args: argparse.Namespace) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_estimates(report_dir / "agent_estimates.csv", estimates)
    aggregate_rows = aggregate(estimates)
    _write_rows(report_dir / "aggregate_progress.csv", aggregate_rows)
    _plot(report_dir / "aggregate_progress.png", aggregate_rows)
    _write_readme(report_dir, row, args, aggregate_rows)


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


def _plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["turn"]) for row in rows]
    y = [float(row["mean_fraction_complete"]) for row in rows]
    lo = [float(row["lower_1sd"]) for row in rows]
    hi = [float(row["upper_1sd"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.fill_between(x, lo, hi, alpha=0.2, color="#59A14F", linewidth=0)
    axis.plot(x, y, color="#2E6B2E", linewidth=2)
    axis.scatter(x, y, s=16, color="#2E6B2E")
    axis.set_ylim(0, 1)
    axis.set_xlabel("turn")
    axis.set_ylabel("mean fraction complete")
    axis.set_title("Together 40-agent replay ensemble")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_readme(report_dir: Path, row: dict[str, Any], args: argparse.Namespace, aggregate_rows: list[dict[str, Any]]) -> None:
    final = aggregate_rows[-1]
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Together Replay Ensemble",
                "",
                "Blind turn-by-turn ensemble estimates for a solved SWE-Agent trace.",
                "",
                f"- model: `{args.model}`",
                f"- raw row index: `{args.raw_index}`",
                f"- instance: `{row.get('instance_id', '')}`",
                f"- agents: {args.agents}",
                f"- parallel workers per turn: {args.workers}",
                f"- final mean fraction complete: {float(final['mean_fraction_complete']):.3f}",
                f"- final one-sigma band: [{float(final['lower_1sd']):.3f}, {float(final['upper_1sd']):.3f}]",
                f"- dataset target label: `{row.get('target', '')}`",
                "",
                "The target label and evaluator logs were used only for selecting and documenting the solved trace, not in the per-turn prompts.",
                "",
                "Artifacts:",
                "",
                "- `agent_estimates.csv`: one scalar estimate per agent per turn.",
                "- `aggregate_progress.csv`: mean, standard deviation, and one-sigma band by turn.",
                "- `aggregate_progress.png`: aggregate progress curve.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
