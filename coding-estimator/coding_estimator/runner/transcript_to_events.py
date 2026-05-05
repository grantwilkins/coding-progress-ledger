"""Convert subagent transcript JSONL into upstream wire-format events.

Mirrors `coding-progress-ledger/scripts/auto_annotate_hermes.py` so the
output `events.jsonl` is interoperable with the upstream sidecar.

Transcript line schema (emitted by the subagent per the runner prompt):

    {"step": <int>, "ts": "<ISO-8601 UTC>",
     "kind": "shell" | "read_file" | "write_file" | "edit_file" |
             "list_dir" | "grep" | "thought" | "done",
     "summary": "<short imperative description>",
     "command": "<shell command, optional>",
     "path": "<path, optional>",
     "exit_code": <int, optional>,
     "obs_snippet": "<optional observation snippet, <=500 chars>"}

A `done` line closes the run; lines after it are dropped.

Wire-event schema (consumed by ledger_progress.sidecar.LedgerSidecar):

    {"schema_version": "1.0",
     "run_id": "<basename of run_dir>",
     "step": <int>,
     "timestamp": "<ISO-8601 UTC>",
     "ledger_ops": [<add or complete or block op>]}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

INSTALL_RE = re.compile(
    r"\b(pip\s+install|apt-get\s+install|apt\s+install|"
    r"conda\s+install|npm\s+install|uv\s+(add|sync|pip))\b"
)
TEST_RE = re.compile(
    r"\b(pytest|unittest|nosetests|python\s+-m\s+unittest|jest|"
    r"go\s+test|cargo\s+test|npm\s+test)\b"
)
INVEST_SHELL_RE = re.compile(
    r"^\s*(ls|cat|head|tail|pwd|grep|find|which|wc|tree|stat|file|less|more)\b"
)
PRODUCT_SHELL_RE = re.compile(r">>?\s*\S|tee\s+\S|sed\s+-i|>\s*\S")

CATEGORIES = ("investigation", "product", "validation", "environment", "artifact")
KIND_TO_CATEGORY = {
    "read_file": "investigation",
    "list_dir": "investigation",
    "grep": "investigation",
    "find": "investigation",
    "write_file": "product",
    "edit_file": "product",
    "create_file": "product",
}


def classify(line: dict) -> str:
    """Return the upstream SubtaskCategory wire value (lowercase string)."""
    kind = line.get("kind")
    if kind in KIND_TO_CATEGORY:
        return KIND_TO_CATEGORY[kind]
    if kind != "shell":
        raise ValueError(f"unclassifiable transcript kind: {kind!r}")
    cmd = (line.get("command") or "").strip()
    if INSTALL_RE.search(cmd):
        return "environment"
    if TEST_RE.search(cmd):
        return "validation"
    if INVEST_SHELL_RE.match(cmd):
        return "investigation"
    if PRODUCT_SHELL_RE.search(cmd):
        return "product"
    return "investigation"


@dataclass(frozen=True)
class _Group:
    category: str
    lines: tuple[dict, ...]


def _is_actionable(line: dict) -> bool:
    return line.get("kind") not in (None, "thought", "done")


def _actionable_until_done(lines: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ln in lines:
        if ln.get("kind") == "done":
            break
        if _is_actionable(ln):
            out.append(ln)
    return out


def _group(lines: list[dict]) -> list[_Group]:
    groups: list[_Group] = []
    bucket: list[dict] = []
    cur_cat: str | None = None
    for ln in lines:
        cat = classify(ln)
        if cur_cat is None or cat != cur_cat:
            if bucket:
                groups.append(_Group(cur_cat, tuple(bucket)))
            bucket = [ln]
            cur_cat = cat
        else:
            bucket.append(ln)
    if bucket:
        groups.append(_Group(cur_cat, tuple(bucket)))
    return groups


def _evidence(ln: dict) -> str:
    parts = [f"step {ln['step']}"]
    if ln.get("kind") == "shell" and ln.get("command"):
        parts.append(f"shell: {ln['command'][:80]}")
    elif ln.get("path"):
        parts.append(f"{ln['kind']}: {ln['path']}")
    elif ln.get("summary"):
        parts.append(ln["summary"][:80])
    return " ".join(parts)


def _description(group: _Group) -> str:
    first = group.lines[0]
    summary = first.get("summary") or first.get("command") or first.get("path") or ""
    return f"{group.category}: {summary[:120]}"


def transcript_to_events(transcript_lines: list[dict], run_id: str) -> list[dict]:
    """Convert a parsed transcript into a list of wire-format events.

    Deterministic. Produces one (add, complete) pair per category-group.
    """
    actionable = _actionable_until_done(transcript_lines)
    if not actionable:
        return []
    groups = _group(actionable)
    events: list[dict] = []
    for i, g in enumerate(groups, start=1):
        sid = f"S{i}"
        first = g.lines[0]
        last = g.lines[-1]
        events.append({
            "schema_version": "1.0",
            "run_id": run_id,
            "step": first["step"],
            "timestamp": first["ts"],
            "ledger_ops": [{
                "op": "add",
                "step": first["step"],
                "id": sid,
                "category": g.category,
                "description": _description(g),
            }],
        })
        events.append({
            "schema_version": "1.0",
            "run_id": run_id,
            "step": last["step"],
            "timestamp": last["ts"],
            "ledger_ops": [{
                "op": "complete",
                "step": last["step"],
                "id": sid,
                "evidence": _evidence(last),
            }],
        })
    return events


def read_transcript(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"transcript not found: {path}")
    out: list[dict] = []
    for n, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        line = json.loads(raw)
        if not isinstance(line, dict):
            raise ValueError(f"transcript line {n}: expected object, got {type(line).__name__}")
        if "step" not in line or "ts" not in line or "kind" not in line:
            raise ValueError(f"transcript line {n}: missing required field (step/ts/kind)")
        out.append(line)
    return out


def write_events(events: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(e, separators=(",", ":"), sort_keys=True) + "\n" for e in events)
    )
