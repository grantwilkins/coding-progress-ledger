#!/usr/bin/env python3
"""HP5 deterministic heuristic auto-annotator.

Reads `normalized_trace.json` per pilot, emits `ledger.jsonl` by:
  - mapping each assistant tool call to a SubtaskCategory,
  - grouping consecutive same-category assistant steps into ONE leaf,
  - completing leaves whose last paired tool response has no error,
  - blocking leaves on 3+ identical tool-response bodies (Pitfall H3)
    or an explicit `"error"` key in observation,
  - closing an ARTIFACT leaf on a terminal tool call match.
Evidence cites only action / tool_name + args summary; never thought.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger_progress import LedgerSession, SubtaskCategory  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

TERMINAL_TOOL_RE = re.compile(
    r"^(submit_answer|final_response|task_complete|finish|skill_manage|skill_view)$"
)

INVESTIGATION_TOOLS = {
    "read_file", "list_directory", "find_file", "grep", "search",
    "web_search", "fetch_url", "browse",
}
PRODUCT_TOOLS = {"write_file", "patch", "edit_file", "create_file"}
VALIDATION_TOOLS = {"run_tests", "pytest"}
ENVIRONMENT_TOOLS = {"pip", "pip_install", "apt_get"}
ARTIFACT_TOOLS = {"submit_answer", "final_response", "task_complete", "finish",
                  "skill_manage", "skill_view"}

INSTALL_RE = re.compile(r"\b(pip\s+install|apt-get\s+install|apt\s+install|conda\s+install|npm\s+install|uv\s+(add|sync|pip))\b")
TEST_RE = re.compile(r"\b(pytest|unittest|nosetests|jest|go\s+test|cargo\s+test|npm\s+test)\b")
INVEST_SHELL_RE = re.compile(r"^\s*(ls|cat|head|tail|pwd|grep|find|which|wc|tree|stat|file|less|more)\b")
PRODUCT_SHELL_RE = re.compile(r">>?\s*\S|tee\s+\S|sed\s+-i|>\s*\S")


def _classify_shell(args_json: Optional[str]) -> SubtaskCategory:
    cmd = ""
    if args_json:
        try:
            obj = json.loads(args_json)
            if isinstance(obj, dict):
                cmd = obj.get("command") or obj.get("cmd") or obj.get("script") or ""
        except json.JSONDecodeError:
            cmd = args_json
    cmd_l = cmd.strip()
    if INSTALL_RE.search(cmd_l):
        return SubtaskCategory.ENVIRONMENT
    if TEST_RE.search(cmd_l):
        return SubtaskCategory.VALIDATION
    if INVEST_SHELL_RE.match(cmd_l):
        return SubtaskCategory.INVESTIGATION
    if PRODUCT_SHELL_RE.search(cmd_l):
        return SubtaskCategory.PRODUCT
    return SubtaskCategory.VALIDATION


def classify_step(tool_name: Optional[str], args_json: Optional[str]) -> SubtaskCategory:
    if not tool_name:
        return SubtaskCategory.INVESTIGATION
    name = tool_name.strip()
    if name in INVESTIGATION_TOOLS:
        return SubtaskCategory.INVESTIGATION
    if name in PRODUCT_TOOLS:
        return SubtaskCategory.PRODUCT
    if name in VALIDATION_TOOLS:
        return SubtaskCategory.VALIDATION
    if name in ENVIRONMENT_TOOLS:
        return SubtaskCategory.ENVIRONMENT
    if name in ARTIFACT_TOOLS:
        return SubtaskCategory.ARTIFACT
    if name in ("bash", "shell", "terminal"):
        return _classify_shell(args_json)
    return SubtaskCategory.INVESTIGATION


def _args_summary(args_json: Optional[str]) -> str:
    if not args_json:
        return ""
    try:
        obj = json.loads(args_json)
    except json.JSONDecodeError:
        return args_json[:60]
    if isinstance(obj, dict):
        for k in ("path", "file", "command", "cmd", "name", "query", "url"):
            v = obj.get(k)
            if isinstance(v, str):
                return f"{k}={v[:60]}"
    return json.dumps(obj, sort_keys=True)[:60]


def _evidence(step: int, tool_name: Optional[str], args_json: Optional[str]) -> str:
    return f"step {step}: {tool_name or '(no_tool)'}({_args_summary(args_json)})"


def _is_error_response(observation: Optional[str]) -> bool:
    if not observation:
        return False
    try:
        obj = json.loads(observation)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    if "error" in obj and obj["error"] not in (None, "", False):
        return True
    if obj.get("success") is False:
        return True
    if isinstance(obj.get("exit_code"), int) and obj["exit_code"] != 0:
        return True
    return False


def _response_body(observation: Optional[str]) -> str:
    if not observation:
        return ""
    try:
        obj = json.loads(observation)
    except json.JSONDecodeError:
        return observation
    if isinstance(obj, dict):
        for k in ("content", "output", "result", "message"):
            v = obj.get(k)
            if isinstance(v, str):
                return v
        return json.dumps(obj, sort_keys=True)
    return str(obj)


def _paired_response(events: List[Dict[str, Any]], step_idx: int) -> Optional[Dict[str, Any]]:
    for ev in events:
        if ev["role"] != "tool":
            continue
        raw = ev.get("raw") or {}
        if raw.get("paired_call_event_step") == step_idx:
            return ev
    return None


def _build_groups(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for ev in events:
        if ev["role"] != "assistant" or ev.get("action") is None:
            continue
        cat = classify_step(ev.get("tool_name"), ev.get("command"))
        is_terminal = bool(ev.get("tool_name") and TERMINAL_TOOL_RE.match(ev["tool_name"]))
        if is_terminal:
            cat = SubtaskCategory.ARTIFACT
        if cur is None or cur["category"] != cat:
            cur = {"category": cat, "steps": []}
            groups.append(cur)
        cur["steps"].append(ev["step_index"])
    return groups


def auto_annotate(normalized: Dict[str, Any]) -> LedgerSession:
    events = normalized["events"]
    by_idx = {e["step_index"]: e for e in events}
    issue = normalized.get("issue_text") or "(no task description)"
    s = LedgerSession(issue.splitlines()[0][:200] if issue else "hermes auto task")
    groups = _build_groups(events)
    leaf_counter = 0
    for g in groups:
        steps = g["steps"]
        cat = g["category"]
        first_step = steps[0]
        leaf_counter += 1
        sid = f"S{leaf_counter}"
        first_ev = by_idx[first_step]
        desc_summary = _args_summary(first_ev.get("command"))
        desc = f"{cat.name.lower()} via {first_ev.get('tool_name')} ({desc_summary})"
        s.add(desc, step=first_step, category=cat)
        bodies: List[str] = []
        blocked = False
        block_step = first_step
        block_reason = ""
        block_evidence_step = first_step
        last_complete_step = first_step
        last_complete_evidence: Optional[str] = None
        for step_idx in steps:
            ev = by_idx[step_idx]
            resp = _paired_response(events, step_idx)
            if resp is None:
                continue
            obs = resp.get("observation")
            body = _response_body(obs)
            bodies.append(body)
            if _is_error_response(obs):
                blocked = True
                block_step = resp["step_index"]
                block_reason = "error response"
                block_evidence_step = step_idx
                break
            if len(bodies) >= 3 and bodies[-1] == bodies[-2] == bodies[-3]:
                blocked = True
                block_step = resp["step_index"]
                block_reason = "3+ identical tool responses (Pitfall H3)"
                block_evidence_step = step_idx
                break
            last_complete_step = resp["step_index"]
            last_complete_evidence = _evidence(step_idx, ev.get("tool_name"), ev.get("command"))
        if blocked:
            ev = by_idx[block_evidence_step]
            s.block(sid, step=block_step, reason=block_reason,
                    evidence=_evidence(block_evidence_step, ev.get("tool_name"), ev.get("command")))
        elif last_complete_evidence is not None:
            s.complete(sid, last_complete_evidence, step=last_complete_step)
    return s


def emit_one(run_dir: Path) -> Dict[str, Any]:
    norm_path = run_dir / "normalized_trace.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"missing {norm_path}")
    normalized = json.loads(norm_path.read_text(encoding="utf-8"))
    session = auto_annotate(normalized)
    session.export_jsonl(str(run_dir / "ledger.jsonl"))
    subprocess.run(
        ["uv", "run", "ledger-run", "export-run", str(run_dir)],
        check=True, cwd=str(REPO_ROOT),
    )
    quality = {
        "annotator": "auto_annotate_hermes (HP5 heuristic)",
        "annotation_pass": "heuristic-auto",
        "annotation_time_minutes": 0,
        "number_of_uncertain_events": 0,
        "number_of_evidence_gaps": 0,
        "whether_final_success_used_only_at_end": True,
        "whether_progress_forced": False,
        "whether_schema_gap_found": False,
        "number_of_subtasks": len(session.ledger.subtasks),
    }
    (run_dir / "annotation_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {"run_dir": str(run_dir), "n_leaves": len(session.ledger.subtasks)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HP5 heuristic auto-annotator.")
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--only", action="append", default=None)
    args = p.parse_args(argv)
    pilot_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if args.only:
        wanted = set(args.only)
        pilot_dirs = [d for d in pilot_dirs if d.name in wanted]
    if not pilot_dirs:
        print(f"[auto_annotate_hermes] FATAL: no pilot dirs under {args.runs_dir}", file=sys.stderr)
        return 2
    for d in pilot_dirs:
        info = emit_one(d)
        print(f"[auto_annotate_hermes] {d.name}: {info['n_leaves']} leaves", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
