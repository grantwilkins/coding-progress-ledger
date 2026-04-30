#!/usr/bin/env python3
"""SWE-agent trace normalizer (Workstream C, task C2).

Reads one raw row (already on disk as JSON) and emits:

  * <run_dir>/normalized_trace.json   — the C1 schema
  * <run_dir>/trajectory_summary.md   — a short human summary

The raw row is NEVER modified. C3 is responsible for keeping the raw
artifact at <run_dir>/source_trace.json; this script only reads from
the path supplied via ``--source-row``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = 1
KNOWN_TOP_LEVEL_KEYS = (
    "instance_id",
    "model_name",
    "target",
    "trajectory",
    "exit_status",
    "generated_patch",
    "eval_logs",
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize one SWE-agent trace row.")
    p.add_argument("--source-row", required=True, type=Path, help="Path to a JSON file containing one decoded upstream row.")
    p.add_argument("--run-dir", required=True, type=Path, help="Destination run directory (will be created).")
    p.add_argument("--source", default="swe_agent_nebius", choices=("swe_agent_nebius",))
    return p.parse_args(argv)


def _split_thought_and_command(text: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Split an assistant-turn text into (thought, command, warnings).

    Looks for the FIRST triple-backtick fence and its closing fence.
    Anything before the opening fence is `thought`; the literal text
    inside the fences is `command`. If parsing fails, the whole text
    becomes the thought and a warning string is appended.
    """
    warnings: List[str] = []
    if text is None:
        return None, None, ["text_is_none"]
    open_idx = text.find("```")
    if open_idx == -1:
        return (text.strip() or None), None, ["no_fenced_block"]
    after_open = open_idx + 3
    # Skip an optional language tag on the same line as the opening fence.
    nl_after_open = text.find("\n", after_open)
    if nl_after_open == -1:
        return (text[:open_idx].strip() or None), None, ["fence_unterminated"]
    body_start = nl_after_open + 1
    close_idx = text.find("```", body_start)
    if close_idx == -1:
        return (text[:open_idx].strip() or None), None, ["fence_unterminated"]
    thought = text[:open_idx].strip() or None
    command = text[body_start:close_idx].strip() or None
    return thought, command, warnings


def _internal_role(upstream_role: Any, is_first_non_system: bool) -> str:
    if upstream_role == "system":
        return "system"
    if upstream_role == "ai":
        return "assistant"
    if upstream_role == "user":
        return "environment" if is_first_non_system else "tool"
    return "unknown"


def _normalize_event(
    raw: Dict[str, Any],
    step_index: int,
    is_first_non_system: bool,
    prior_command: Optional[str],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Returns (normalized_event, command_for_next_step)."""
    upstream_role = raw.get("role")
    role = _internal_role(upstream_role, is_first_non_system)
    text = raw.get("text") if isinstance(raw.get("text"), str) or raw.get("text") is None else str(raw.get("text"))

    thought = action = observation = command = tool_name = None
    parse_warnings: List[str] = []

    if role == "assistant":
        thought, command, parse_warnings = _split_thought_and_command(text)
        action = command
        if command:
            tool_name = command.split(None, 1)[0]
        next_command = command
    elif role == "tool":
        observation = text
        if prior_command:
            tool_name = prior_command.split(None, 1)[0]
        next_command = None
    elif role == "environment":
        observation = text
        next_command = None
    else:
        # system or unknown: leave fields as None
        next_command = None

    raw_with_warnings: Dict[str, Any] = dict(raw)
    if parse_warnings:
        raw_with_warnings["parse_warnings"] = parse_warnings

    event = {
        "step_index": step_index,
        "role": role,
        "thought": thought,
        "action": action,
        "observation": observation,
        "tool_name": tool_name,
        "command": command,
        "files_touched": [],
        "timestamp": None,
        "raw": raw_with_warnings,
    }
    return event, next_command


def normalize_row(row: Dict[str, Any], source: str = "swe_agent_nebius") -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"row must be a dict, got {type(row).__name__}")

    trajectory = row.get("trajectory")
    if trajectory is None:
        trajectory = []
    if not isinstance(trajectory, list):
        raise TypeError(f"row['trajectory'] must be a list when present, got {type(trajectory).__name__}")

    target = row.get("target")
    final_success = target if isinstance(target, bool) else None

    events: List[Dict[str, Any]] = []
    seen_non_system = False
    prior_command: Optional[str] = None
    system_prompt: Optional[str] = None
    issue_text: Optional[str] = None

    for i, raw in enumerate(trajectory):
        if not isinstance(raw, dict):
            raw = {"role": "unknown", "text": None, "_original_type": type(raw).__name__}
        is_first_non_system = (raw.get("role") != "system") and (not seen_non_system)
        event, next_command = _normalize_event(
            raw, step_index=i, is_first_non_system=is_first_non_system, prior_command=prior_command
        )
        if event["role"] != "system":
            seen_non_system = True
        if event["role"] == "system" and system_prompt is None:
            sp = raw.get("system_prompt")
            if isinstance(sp, str) and sp:
                system_prompt = sp
        if is_first_non_system and event["role"] == "environment":
            obs = event["observation"]
            if isinstance(obs, str) and obs:
                issue_text = obs
        if event["role"] == "assistant":
            prior_command = next_command
        events.append(event)

    extra = sorted(k for k in row.keys() if k not in KNOWN_TOP_LEVEL_KEYS)

    patch = row.get("generated_patch")
    eval_logs = row.get("eval_logs")
    raw_metadata = {
        "patch_length": len(patch) if isinstance(patch, str) else 0,
        "eval_logs_length": len(eval_logs) if isinstance(eval_logs, str) else 0,
        "extra_top_level_keys": extra,
    }

    instance_id = row.get("instance_id") if isinstance(row.get("instance_id"), str) else ""
    model_name = row.get("model_name") if isinstance(row.get("model_name"), str) else ""
    exit_status = row.get("exit_status") if isinstance(row.get("exit_status"), str) else None

    out = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "instance_id": instance_id,
        "model_name": model_name,
        "exit_status": exit_status,
        "final_success": final_success,
        "trajectory_length": len(events),
        "issue_text": issue_text,
        "system_prompt": system_prompt,
        "events": events,
        "raw_metadata": raw_metadata,
    }
    assert out["trajectory_length"] == len(out["events"])
    return out


def render_summary(norm: Dict[str, Any], max_steps: int = 60) -> str:
    lines: List[str] = []
    lines.append(f"# Trajectory summary — {norm['instance_id']}")
    lines.append("")
    lines.append(f"- model: `{norm['model_name']}`")
    lines.append(f"- final_success: `{norm['final_success']}`")
    lines.append(f"- exit_status: `{norm['exit_status']}`")
    lines.append(f"- trajectory_length: `{norm['trajectory_length']}`")
    lines.append(f"- patch_length: `{norm['raw_metadata']['patch_length']}`")
    lines.append(f"- eval_logs_length: `{norm['raw_metadata']['eval_logs_length']}`")
    lines.append("")
    if norm["issue_text"]:
        head = norm["issue_text"].splitlines()[0][:200]
        lines.append(f"**Issue:** {head}")
        lines.append("")
    lines.append("## Steps")
    lines.append("")
    for ev in norm["events"][:max_steps]:
        tag = ev["role"]
        if ev["role"] == "assistant":
            cmd = ev["command"] or "(no command)"
            cmd_short = cmd.splitlines()[0][:120]
            lines.append(f"- [{ev['step_index']:03d}] {tag}: `{cmd_short}`")
        elif ev["role"] in ("tool", "environment"):
            obs = (ev["observation"] or "").splitlines()
            head = obs[0][:120] if obs else "(empty)"
            lines.append(f"- [{ev['step_index']:03d}] {tag}: {head}")
        else:
            lines.append(f"- [{ev['step_index']:03d}] {tag}")
    if len(norm["events"]) > max_steps:
        lines.append(f"- ... ({len(norm['events']) - max_steps} more steps elided)")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    row = json.loads(args.source_row.read_text(encoding="utf-8"))
    norm = normalize_row(row, source=args.source)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "normalized_trace.json").write_text(
        json.dumps(norm, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "trajectory_summary.md").write_text(
        render_summary(norm), encoding="utf-8"
    )
    print(
        f"[normalize_swe_agent_trace] wrote {args.run_dir / 'normalized_trace.json'} "
        f"({norm['trajectory_length']} events)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
