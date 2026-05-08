from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import utc_now
from .protocol import VERSIONS


VALIDATION_RE = re.compile(
    r"(\bpytest\b|\bunittest\b|\bnpm\s+test\b|\bgo\s+test\b|\bcargo\s+test\b|"
    r"\brspec\b|\btox\b|\bmake\s+test\b|run-tests\.sh|/tests?\b|"
    r"validation|validate|\bpy_compile\b|\bfinal[-_]check\b|"
    r"\bexample matches:\s*true\b|\brunning basic .{0,80}\btest\b|"
    r"\b(completed|processed) successfully\b|\b(cmp|diff)\b|"
    r"\b(pip|pip3)\s+--version\b|"
    r"\bpython[0-9.]*\s+-m\s+pip\s+--version\b|"
    r"\b(pip|pip3)\b.*(?:^|\s)--dry-run\b)",
    re.IGNORECASE,
)


def transcript_to_wire_events(
    transcript: list[dict[str, Any]],
    *,
    run_id: str,
    verifier_exit_code: int | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    completed_leaf_ids: list[str] = []
    for row in transcript:
        step = int(row.get("step", len(events) + 1))
        kind = str(row.get("kind") or "unknown")
        description = str(row.get("summary") or row.get("command") or kind)
        category = _category_for_row(row)
        subtask_id = f"{category}_{step}"
        ledger_ops = row.get("ledger_ops")
        if not ledger_ops:
            ledger_ops = _inferred_ledger_ops(row, subtask_id=subtask_id, description=description, category=category)
        _record_completed_ids(ledger_ops, completed_leaf_ids)
        events.append(
            {
                "schema_version": VERSIONS.ledger_wire_schema_version,
                "run_id": run_id,
                "step": step,
                "timestamp": row.get("ts") or row.get("timestamp") or utc_now(),
                "agent_step": row,
                "ledger_ops": ledger_ops,
            }
        )
    if verifier_exit_code is not None:
        terminal_step = max((int(row.get("step", 0)) for row in transcript), default=0) + 1
        terminal_ops = _terminal_verifier_ops(verifier_exit_code, completed_leaf_ids=completed_leaf_ids)
        events.append(
            {
                "schema_version": VERSIONS.ledger_wire_schema_version,
                "run_id": run_id,
                "step": terminal_step,
                "timestamp": utc_now(),
                "agent_step": {
                    "kind": "terminal_verifier_result",
                    "summary": "final verifier passed" if verifier_exit_code == 0 else "final verifier failed",
                    "visible_to_agent": False,
                },
                "ledger_ops": terminal_ops,
            }
        )
    return events


def write_wire_events(path: Path, events: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def replay_sidecar(
    *,
    run_dir: Path,
    ledger_root: Path,
    input_file: Path | None = None,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        python_executable or sys.executable,
        "-m",
        "ledger_progress.sidecar",
        "--run-dir",
        str(run_dir.resolve()),
        "--input-file",
        str((input_file or run_dir / "events.jsonl").resolve()),
    ]
    return subprocess.run(command, cwd=ledger_root, capture_output=True, text=True)


def _category_for_row(row: dict[str, Any]) -> str:
    text = _row_text(row)
    lowered = text.lower()
    if VALIDATION_RE.search(text) or any(token in lowered for token in ("pytest", "test", "check", "verify", "lint")):
        return "validation"
    if row.get("kind") in {"write_file", "edit_file", "patch", "apply_patch"}:
        return "product"
    if any(token in lowered for token in ("pip install", "apt", "docker", "venv")):
        return "environment"
    return "investigation"


def _inferred_ledger_ops(
    row: dict[str, Any],
    *,
    subtask_id: str,
    description: str,
    category: str,
) -> list[dict[str, Any]]:
    add_op = {
        "op": "add",
        "id": subtask_id,
        "description": description[:240],
        "category": category,
    }
    exit_code = row.get("exit_code")
    evidence = _evidence(row, description=description)
    if row.get("kind") == "done":
        return [
            add_op,
            {
                "op": "complete",
                "id": subtask_id,
                "evidence": evidence,
            },
        ]
    if row.get("kind") in {"read_file", "list_dir", "find_files", "grep", "write_file", "edit_file", "patch", "apply_patch"}:
        if exit_code == 0:
            return [
                add_op,
                {
                    "op": "complete",
                    "id": subtask_id,
                    "evidence": evidence,
                },
            ]
    if row.get("kind") == "shell":
        if exit_code == 0:
            return [
                add_op,
                {
                    "op": "complete",
                    "id": subtask_id,
                    "evidence": evidence,
                },
            ]
        if exit_code not in (None, 0):
            return [
                add_op,
                {
                    "op": "block",
                    "id": subtask_id,
                    "reason": _failure_reason(row),
                    "evidence": evidence,
                },
            ]
    if row.get("kind") in {
        "dependency_missing",
        "network_blocked",
        "environment_blocked",
        "tool_denied",
        "early_done_denied",
        "model_parse_error",
        "invalid_action",
    }:
        return [
            add_op,
            {
                "op": "block",
                "id": subtask_id,
                "reason": description[:240] or "tool/controller event blocked progress",
            },
        ]
    return [
        add_op,
        {
            "op": "start",
            "id": subtask_id,
            "reason": "transcript-derived work observed; completion requires explicit ledger_ops or done boundary",
        },
    ]


def _terminal_verifier_ops(verifier_exit_code: int, *, completed_leaf_ids: list[str]) -> list[dict[str, Any]]:
    if verifier_exit_code == 0:
        return []
    terminal_id = "terminal_verifier"
    ops: list[dict[str, Any]] = [
        {
            "op": "add",
            "id": terminal_id,
            "description": "Run final hidden verifier",
            "category": "validation",
        },
        {
            "op": "block",
            "id": terminal_id,
            "reason": "Final verifier returned nonzero exit code.",
            "evidence": "Final verifier failed after the agent-visible phase.",
        },
    ]
    if completed_leaf_ids:
        ops.append(
            {
                "op": "reopen",
                "id": completed_leaf_ids[-1],
                "reason": "Final verifier failure showed prior visible work was incomplete.",
            }
        )
    return ops


def _record_completed_ids(ledger_ops: list[dict[str, Any]], completed_leaf_ids: list[str]) -> None:
    for op in ledger_ops:
        if op.get("op") == "complete" and isinstance(op.get("id"), str):
            completed_leaf_ids.append(op["id"])
        elif op.get("op") == "reopen" and op.get("id") in completed_leaf_ids:
            completed_leaf_ids.remove(op["id"])


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("kind", "summary", "command", "path", "stdout_snippet", "stderr_snippet", "obs_snippet")
    )


def _evidence(row: dict[str, Any], *, description: str) -> str:
    text = _row_text(row).strip() or description
    return text[:240]


def _failure_reason(row: dict[str, Any]) -> str:
    text = _row_text(row).strip()
    return (text[:240] if text else "shell command returned nonzero exit code")
