#!/usr/bin/env python3
"""Hermes trace normalizer (HP2). Mirrors normalize_swe_agent_trace.py."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = 1
KNOWN_TOP_LEVEL_KEYS = ("id", "task", "tools", "category", "subcategory", "conversations")
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
PATH_TOOLS = {"write_file", "patch", "edit_file", "read_file"}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize one Hermes trace row.")
    p.add_argument("--source-row", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--source", default="hermes_agent_reasoning", choices=("hermes_agent_reasoning",))
    p.add_argument("--model-name", default="", help="Hermes config (kimi or glm-5.1).")
    return p.parse_args(argv)


def _strip_think(text: str) -> Tuple[str, str]:
    """Return (collapsed_thought, remainder_without_think_blocks)."""
    thoughts = [m.group(1).strip() for m in THINK_RE.finditer(text)]
    remainder = THINK_RE.sub("", text)
    return ("\n".join(t for t in thoughts if t).strip(), remainder)


def _parse_tool_call(block: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Parse a <tool_call> JSON body into (name, args_json, files_touched)."""
    obj = json.loads(block.strip())
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    args = obj.get("arguments")
    args_json = json.dumps(args, ensure_ascii=False, sort_keys=True) if args is not None else None
    files: List[str] = []
    if name in PATH_TOOLS and isinstance(args, dict):
        for k in ("path", "file"):
            if isinstance(args.get(k), str):
                files.append(args[k])
                break
    return name, args_json, files


def _parse_tool_response(block: str) -> Tuple[Optional[str], Optional[str], str]:
    """Parse a <tool_response> JSON body into (tool_call_id, name, content_text)."""
    obj = json.loads(block.strip())
    tcid = obj.get("tool_call_id") if isinstance(obj.get("tool_call_id"), str) else None
    name = obj.get("name") if isinstance(obj.get("name"), str) else None
    content = obj.get("content")
    content_text = json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content
    return tcid, name, content_text


def _split_assistant_turn(value: str) -> Tuple[str, str, List[Tuple[str, Dict[str, Any]]]]:
    """Return (thought, free_text_after_think, [(call_block_text, {name,args_json,files,id})...])."""
    thought, remainder = _strip_think(value)
    calls: List[Tuple[str, Dict[str, Any]]] = []
    free_text_parts: List[str] = []
    pos = 0
    for m in TOOL_CALL_RE.finditer(remainder):
        free_text_parts.append(remainder[pos:m.start()])
        block = m.group(1)
        name, args_json, files = _parse_tool_call(block)
        obj = json.loads(block.strip())
        cid = obj.get("id") if isinstance(obj.get("id"), str) else None
        calls.append((block, {"name": name, "args_json": args_json, "files": files, "id": cid}))
        pos = m.end()
    free_text_parts.append(remainder[pos:])
    free_text = "\n".join(p.strip() for p in free_text_parts if p.strip()).strip()
    return thought, free_text, calls


def _split_tool_turn(value: str) -> List[Tuple[str, str, str]]:
    """Return list of (tool_call_id, name, content_text) per <tool_response>."""
    out: List[Tuple[str, str, str]] = []
    for m in TOOL_RESPONSE_RE.finditer(value):
        out.append(_parse_tool_response(m.group(1)))
    return out


def _empty_event(step_index: int, role: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_index": step_index,
        "role": role,
        "thought": None,
        "action": None,
        "observation": None,
        "tool_name": None,
        "command": None,
        "files_touched": [],
        "timestamp": None,
        "raw": raw,
    }


def _pair_responses(call_meta: List[Dict[str, Any]], responses: List[Tuple[str, str, str]]) -> List[Optional[Tuple[str, str, str]]]:
    """For each call, find a matching response by tool_call_id, else positional."""
    used = [False] * len(responses)
    paired: List[Optional[Tuple[str, str, str]]] = [None] * len(call_meta)
    for i, c in enumerate(call_meta):
        cid = c.get("id")
        if cid:
            for j, r in enumerate(responses):
                if not used[j] and r[0] == cid:
                    paired[i] = r
                    used[j] = True
                    break
    for i in range(len(call_meta)):
        if paired[i] is None:
            for j in range(len(responses)):
                if not used[j]:
                    paired[i] = responses[j]
                    used[j] = True
                    break
    return paired


def normalize_row(row: Dict[str, Any], source: str = "hermes_agent_reasoning", model_name: str = "") -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"row must be a dict, got {type(row).__name__}")
    convs = row.get("conversations")
    if convs is None:
        convs = []
    if not isinstance(convs, list):
        raise TypeError(f"row['conversations'] must be a list when present, got {type(convs).__name__}")

    events: List[Dict[str, Any]] = []
    system_prompt: Optional[str] = None
    pending_calls: List[Dict[str, Any]] = []
    pending_call_event_ids: List[int] = []

    n = len(convs)
    i = 0
    step = 0
    while i < n:
        c = convs[i]
        if not isinstance(c, dict):
            events.append(_empty_event(step, "unknown", {"_original_type": type(c).__name__}))
            step += 1
            i += 1
            continue
        upstream_role = c.get("from")
        role = ROLE_MAP.get(upstream_role, "unknown")
        value = c.get("value") if isinstance(c.get("value"), str) else ""

        if role == "system":
            if system_prompt is None and value:
                system_prompt = value
            ev = _empty_event(step, "system", dict(c))
            events.append(ev)
            step += 1
            i += 1
            continue
        if role == "user":
            ev = _empty_event(step, "user", dict(c))
            ev["observation"] = value
            events.append(ev)
            step += 1
            i += 1
            continue
        if role == "assistant":
            thought, free_text, calls = _split_assistant_turn(value)
            combined_thought_first = "\n\n".join(t for t in (thought, free_text) if t).strip() or None
            if not calls:
                ev = _empty_event(step, "assistant", dict(c))
                ev["thought"] = combined_thought_first
                events.append(ev)
                step += 1
                i += 1
                continue
            assistant_event_ids: List[int] = []
            call_meta_list: List[Dict[str, Any]] = []
            for k, (block, meta) in enumerate(calls):
                ev = _empty_event(step, "assistant", {"from": "gpt", "tool_call_block": block, "tool_call_index_in_turn": k})
                if k == 0:
                    ev["thought"] = combined_thought_first
                ev["action"] = meta["name"]
                ev["tool_name"] = meta["name"]
                ev["command"] = meta["args_json"]
                ev["files_touched"] = list(meta["files"])
                ev["raw"]["tool_call_id"] = meta["id"]
                events.append(ev)
                assistant_event_ids.append(step)
                call_meta_list.append(meta)
                step += 1
            pending_calls = call_meta_list
            pending_call_event_ids = assistant_event_ids
            i += 1
            continue
        if role == "tool":
            responses = _split_tool_turn(value)
            paired = _pair_responses(pending_calls, responses) if pending_calls else []
            for k, (call_meta, ev_idx) in enumerate(zip(pending_calls, pending_call_event_ids)):
                resp = paired[k] if k < len(paired) else None
                tool_ev_raw = {"from": "tool", "tool_response_index": k, "paired_call_event_step": ev_idx}
                if resp is not None:
                    tool_ev_raw["tool_call_id"] = resp[0]
                ev = _empty_event(step, "tool", tool_ev_raw)
                ev["tool_name"] = call_meta.get("name")
                ev["observation"] = resp[2] if resp is not None else None
                events.append(ev)
                step += 1
            unmatched_responses = len(responses) - len(pending_calls)
            for k in range(max(0, unmatched_responses)):
                resp = responses[len(pending_calls) + k]
                ev = _empty_event(step, "tool", {"from": "tool", "tool_response_index": len(pending_calls) + k, "unmatched": True})
                ev["tool_name"] = resp[1]
                ev["observation"] = resp[2]
                events.append(ev)
                step += 1
            pending_calls = []
            pending_call_event_ids = []
            i += 1
            continue
        ev = _empty_event(step, "unknown", dict(c))
        events.append(ev)
        step += 1
        i += 1

    issue_text = None
    for ev in events:
        if ev["role"] == "user" and ev["observation"]:
            issue_text = ev["observation"]
            break
    if issue_text is None and isinstance(row.get("task"), str):
        issue_text = row["task"]

    extra = sorted(k for k in row.keys() if k not in KNOWN_TOP_LEVEL_KEYS)
    tools_field = row.get("tools")
    raw_metadata = {
        "category": row.get("category") if isinstance(row.get("category"), str) else "",
        "subcategory": row.get("subcategory") if isinstance(row.get("subcategory"), str) else "",
        "tool_definitions_length": len(tools_field) if isinstance(tools_field, str) else 0,
        "upstream_conversation_count": len(convs),
        "extra_top_level_keys": extra,
    }

    instance_id = row.get("id") if isinstance(row.get("id"), str) else ""
    out = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "instance_id": instance_id,
        "model_name": model_name,
        "exit_status": None,
        "final_success": None,
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
    lines.append(f"- trajectory_length: `{norm['trajectory_length']}`")
    lines.append(f"- category: `{norm['raw_metadata']['category']}`")
    lines.append(f"- subcategory: `{norm['raw_metadata']['subcategory']}`")
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
            cmd = ev["tool_name"] or "(thought-only)"
            lines.append(f"- [{ev['step_index']:03d}] {tag}: `{cmd}`")
        elif ev["role"] in ("tool", "user"):
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
    norm = normalize_row(row, source=args.source, model_name=args.model_name)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "normalized_trace.json").write_text(
        json.dumps(norm, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "trajectory_summary.md").write_text(render_summary(norm), encoding="utf-8")
    print(
        f"[normalize_hermes_trace] wrote {args.run_dir / 'normalized_trace.json'} ({norm['trajectory_length']} events)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
