from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .hf import iter_hf_rows
from .models import Turn
from .readers import rows_to_turns

TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parents[2] / "reports" / "together_replay_time"
)


@dataclass(frozen=True)
class Estimate:
    turn: int
    seconds_left: float
    confidence_percent: float
    raw: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single Together time-remaining replay on a SWE-Agent trace."
    )
    parser.add_argument("--raw-index", type=int, default=349)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/hf_cache"))
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--api-key-env", default="TOGETHER_API_KEY")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    validate_replay_args(args.model)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    row = swe_agent_row(args.raw_index, args.cache_dir)
    [(_, turns)] = list(rows_to_turns([row], source="swe-agent"))
    estimates = run_replay(
        turns,
        model=args.model,
        api_key=api_key,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        task_prompt=original_task_prompt(row),
        progress=not args.quiet,
    )
    write_report(args.report_dir, estimates, row, args)
    return 0


def validate_replay_args(model: str) -> None:
    if not model.strip():
        raise ValueError("model is required")


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
            text = str(
                item.get("text") or item.get("content") or item.get("message") or ""
            ).strip()
            if text:
                return text
    raise ValueError(
        f"{row.get('instance_id', '<unknown>')}: no original user task prompt found"
    )


def run_replay(
    turns: list[Turn],
    *,
    model: str,
    api_key: str,
    max_retries: int,
    max_tokens: int,
    temperature: float,
    task_prompt: str,
    progress: bool = False,
    complete: Callable[[list[dict[str, str]], str, str, int, int, float], str]
    | None = None,
) -> list[Estimate]:
    complete = complete or together_complete
    transcript: list[str] = []
    estimates: list[Estimate] = []
    for turn in turns:
        transcript.append(turn_evidence(turn))
        raw = _complete_with_retries(
            complete,
            [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": replay_prompt(task_prompt, transcript)},
            ],
            model,
            api_key,
            max_retries,
            max_tokens,
            temperature,
        )
        seconds_left, confidence = parse_time_estimate(raw)
        estimate = Estimate(turn.step, seconds_left, confidence, raw.strip())
        estimates.append(estimate)
        if progress:
            print(
                f"turn={estimate.turn} remaining={estimate.seconds_left:g}s confidence={estimate.confidence_percent:g}%",
                flush=True,
            )
    return estimates


def system_prompt() -> str:
    return (
        "Estimate time remaining for a coding trace from only the original prompt "
        "and observed turns supplied by the user. Return only {XXs, YY%}."
    )


def turn_prompt(task_prompt: str, turn: Turn) -> str:
    return replay_prompt(task_prompt, [turn_evidence(turn)])


def replay_prompt(task_prompt: str, turns_so_far: list[str]) -> str:
    task_prompt = task_prompt.strip()
    if not task_prompt:
        raise ValueError("task_prompt is empty")
    return (
        f"Original prompt:\n{task_prompt}\n\n"
        "State up until this point:\n\n"
        + "\n\n".join(turns_so_far)
        + "\n\n"
        "Given the original prompt and the state up until this point, estimate "
        "the time in seconds left to complete the task specified by this prompt "
        "and give a percentage of your confidence in this estimate. Return only "
        "{XXs, YY%} as your response."
    )


def turn_evidence(turn: Turn) -> str:
    return f"Turn {turn.step}: {turn.kind.upper()}{' via ' + turn.tool if turn.tool else ''}.\n{_turn_text(turn)}"


def parse_time_estimate(text: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"\{\s*(\d+(?:\.\d+)?)\s*s\s*,\s*(\d+(?:\.\d+)?)\s*%\s*\}",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"invalid time estimate response: {text!r}")
    seconds_left = float(match.group(1))
    confidence = float(match.group(2))
    if confidence > 100.0:
        raise ValueError(f"confidence out of range: {confidence}")
    return seconds_left, confidence


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
    estimates: list[Estimate],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_estimates(report_dir / "time_estimates.csv", estimates)
    _plot(report_dir / "remaining_time.png", estimates)
    _write_readme(report_dir, row, args, estimates[-1])


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
            raw = complete(
                messages, model, api_key, max_retries, max_tokens, temperature
            )
            parse_time_estimate(raw)
            return raw
        except ValueError:
            if attempt == max_retries:
                raise
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": "Your previous response was invalid. Reply only in the form {XXs, YY%}, for example {120s, 80%}.",
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


def _plot(path: Path, estimates: list[Estimate]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3.2))
    axis.plot(
        [estimate.turn for estimate in estimates],
        [estimate.seconds_left for estimate in estimates],
        color="#276FBF",
        linewidth=2,
    )
    axis.scatter(
        [estimate.turn for estimate in estimates],
        [estimate.seconds_left for estimate in estimates],
        s=16,
        color="#276FBF",
    )
    axis.set_xlabel("turn")
    axis.set_ylabel("seconds remaining")
    axis.set_title("Together replay time remaining")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_readme(
    report_dir: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    final: Estimate,
) -> None:
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Together Replay Time Remaining",
                "",
                "Single-agent turn-by-turn estimates for one SWE-Agent trace.",
                "",
                f"- model: `{args.model}`",
                f"- raw row index: `{args.raw_index}`",
                f"- instance: `{row.get('instance_id', '')}`",
                f"- final remaining seconds: {final.seconds_left:g}",
                f"- final confidence: {final.confidence_percent:g}%",
                "",
                "Artifacts:",
                "",
                "- `time_estimates.csv`: one time and confidence estimate per turn.",
                "- `remaining_time.png`: seconds-left curve by turn.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
