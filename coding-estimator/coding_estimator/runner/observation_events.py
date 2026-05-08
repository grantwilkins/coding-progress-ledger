"""Structured observation-event emission for tb_live-style runs.

This layer is additive to the ledger. It preserves transcript/verifier
signals that the ledger intentionally discards so downstream models can
consume a less-lossy measurement channel without changing progress
semantics.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"

VALIDATION_RE = re.compile(
    r"\b(pytest|unittest|smoke|verify|verification|test|check|curl|health|version|items|"
    r"python3?\s+-m\s+py_compile|python3?\s+-c|python3?\s+smoke|verifier)\b",
    re.IGNORECASE,
)
PASS_RE = re.compile(r"\b(pass|passed|ok|all checks passed|verified|succeeded)\b", re.IGNORECASE)
FAIL_RE = re.compile(
    r"\b(fail|failed|error|traceback|assertionerror|exception|remote disconnected|"
    r"calledprocesserror|not created|no such file|modulenotfounderror|command not found)\b",
    re.IGNORECASE,
)
ENV_BLOCK_RE = re.compile(
    r"\b(modulenotfounderror|no module named|command not found|no such file or directory|"
    r"permission denied|remote disconnected|connection refused|cannot import)\b",
    re.IGNORECASE,
)
PATH_RE = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)")
ERROR_TOKEN_RE = re.compile(
    r"(AssertionError:[^\n]+|ModuleNotFoundError:[^\n]+|FileNotFoundError:[^\n]+|"
    r"RemoteDisconnected:[^\n]+|CalledProcessError:[^\n]+|No such file or directory[^\n]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObservationEvent:
    run_id: str
    step: int
    observed_ts: str | None
    source_artifact: str
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "step": self.step,
            "observed_ts": self.observed_ts,
            "source_artifact": self.source_artifact,
            "event_type": self.event_type,
            "payload": self.payload,
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_observation_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: list[dict[str, Any]] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{n}: expected JSON object")
        out.append(row)
    return out


def write_observation_events(events: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def _read_transcript(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _text_blob(line: dict[str, Any]) -> str:
    parts = [
        str(line.get("summary") or ""),
        str(line.get("command") or ""),
        str(line.get("obs_snippet") or ""),
        str(line.get("path") or ""),
    ]
    return " | ".join(part for part in parts if part).strip()


def _is_validation_attempt(line: dict[str, Any]) -> bool:
    if line.get("kind") != "shell":
        return False
    return bool(VALIDATION_RE.search(_text_blob(line)))


def _validation_outcome(line: dict[str, Any]) -> str | None:
    text = _text_blob(line)
    exit_code = line.get("exit_code")
    if exit_code == 0 and (PASS_RE.search(text) or _is_validation_attempt(line)):
        return "pass"
    if exit_code not in (None, 0):
        return "fail"
    if FAIL_RE.search(text):
        return "fail"
    if PASS_RE.search(text) and _is_validation_attempt(line):
        return "pass"
    return None


def _normalized_error_signature(text: str) -> str | None:
    match = ERROR_TOKEN_RE.search(text)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip().lower()
    if FAIL_RE.search(text):
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text[:160]
    return None


def _path_parts_from_expr(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_parts_from_expr(node.left)
        right = node.right
        if left is None:
            return None
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return [*left, right.value]
    if isinstance(node, ast.Name) and node.id in {"WS", "APP", "ENV", "OUT", "RUN", "MAKEFILE", "WIDGET", "RENDER"}:
        # Only WS is a true root; others are aliases defined by other
        # assignments and should not be treated as independent roots.
        if node.id == "WS":
            return []
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "cwd"
            and isinstance(func.value, ast.Name)
            and func.value.id == "Path"
        ):
            return []
    return None


def expected_product_paths_for_task(task_id: str | None, repo_root: Path | None = None) -> set[str]:
    if not task_id:
        return set()
    base = repo_root or Path(__file__).resolve().parents[2]
    test_file = base / "tasks" / "tb_live_v2" / task_id / "tests" / "test_outputs.py"
    if not test_file.is_file():
        return set()
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if not target.isupper() or target == "WS":
            continue
        parts = _path_parts_from_expr(node.value)
        if parts is None or not parts:
            continue
        rel_path = "/".join(parts)
        if rel_path in {
            "task.md",
            "task.yaml",
            "shape.yaml",
            "seed.sh",
            "solution.sh",
            "Dockerfile",
            "docker-compose.yaml",
        }:
            continue
        out.add(rel_path)
    return out


def _failure_type(verifier_output: str) -> str:
    lowered = verifier_output.lower()
    if "assertionerror" in lowered or "assert " in lowered:
        return "assertion_failure"
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "module_not_found"
    if "filenotfounderror" in lowered or "no such file or directory" in lowered:
        return "missing_file"
    if "remotedisconnected" in lowered or "connection refused" in lowered:
        return "runtime_disconnect"
    if "calledprocesserror" in lowered:
        return "subprocess_failure"
    return "other"


def build_observation_events(
    *,
    run_dir: Path,
    run_id: str | None = None,
    task_id: str | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    transcript = _read_transcript(run_dir / "transcript.jsonl")
    manifest = _read_json(run_dir / "run_manifest.json")
    verifier_output = (
        (run_dir / "verifier_output.txt").read_text(encoding="utf-8")
        if (run_dir / "verifier_output.txt").is_file()
        else ""
    )
    actual_run_id = run_id or str(run_dir.name)
    actual_task_id = task_id or manifest.get("task_id")
    expected_paths = expected_product_paths_for_task(actual_task_id, repo_root=repo_root)
    workspace_path = run_dir / "workspace_path.txt"
    workspace_root = Path(workspace_path.read_text(encoding="utf-8").strip()) if workspace_path.is_file() else None

    touched_expected: set[str] = set()
    seen_error_signatures: set[str] = set()
    saw_validation = False
    saw_done = False
    saw_oracle = False
    events: list[ObservationEvent] = []

    for line in transcript:
        step = int(line.get("step", 0))
        ts = line.get("ts")
        kind = str(line.get("kind") or "")
        text = _text_blob(line)

        if kind in {"write_file", "edit_file", "create_file"}:
            raw_path = str(line.get("path") or "")
            path_obj = Path(raw_path)
            basename = path_obj.name
            relative_path: str | None = None
            if workspace_root is not None:
                try:
                    anchored = path_obj if path_obj.is_absolute() else (workspace_root / path_obj)
                    relative_path = anchored.resolve().relative_to(workspace_root.resolve()).as_posix()
                except (OSError, ValueError):
                    relative_path = None
            matches_expected = relative_path in expected_paths if relative_path is not None else False
            if matches_expected:
                touched_expected.add(relative_path or "")
            events.append(
                ObservationEvent(
                    run_id=actual_run_id,
                    step=step,
                    observed_ts=ts,
                    source_artifact="transcript.jsonl",
                    event_type="product_file_written",
                    payload={
                        "path": raw_path,
                        "basename": basename,
                        "relative_path": relative_path,
                        "matches_expected_path": matches_expected,
                    },
                )
            )

        if kind == "read_file" and Path(str(line.get("path") or "")).name == "solution.sh":
            saw_oracle = True
            events.append(
                ObservationEvent(
                    run_id=actual_run_id,
                    step=step,
                    observed_ts=ts,
                    source_artifact="transcript.jsonl",
                    event_type="solution_oracle_read",
                    payload={"path": str(line.get("path") or "")},
                )
            )

        if _is_validation_attempt(line):
            saw_validation = True
            events.append(
                ObservationEvent(
                    run_id=actual_run_id,
                    step=step,
                    observed_ts=ts,
                    source_artifact="transcript.jsonl",
                    event_type="validation_attempt",
                    payload={
                        "command": str(line.get("command") or ""),
                        "summary": str(line.get("summary") or ""),
                        "after_solution_oracle_read": saw_oracle,
                    },
                )
            )
            outcome = _validation_outcome(line)
            if outcome == "pass":
                events.append(
                    ObservationEvent(
                        run_id=actual_run_id,
                        step=step,
                        observed_ts=ts,
                        source_artifact="transcript.jsonl",
                        event_type="validation_pass_observed",
                        payload={
                            "command": str(line.get("command") or ""),
                            "summary": str(line.get("summary") or ""),
                        },
                    )
                )
            elif outcome == "fail":
                events.append(
                    ObservationEvent(
                        run_id=actual_run_id,
                        step=step,
                        observed_ts=ts,
                        source_artifact="transcript.jsonl",
                        event_type="validation_fail_observed",
                        payload={
                            "command": str(line.get("command") or ""),
                            "summary": str(line.get("summary") or ""),
                            "exit_code": line.get("exit_code"),
                        },
                    )
                )

        signature = _normalized_error_signature(text)
        exit_code = line.get("exit_code")
        if signature or exit_code not in (None, 0):
            if signature:
                events.append(
                    ObservationEvent(
                        run_id=actual_run_id,
                        step=step,
                        observed_ts=ts,
                        source_artifact="transcript.jsonl",
                        event_type="error_observed",
                        payload={
                            "signature": signature,
                            "exit_code": exit_code,
                            "summary": str(line.get("summary") or ""),
                        },
                    )
                )
                if signature in seen_error_signatures:
                    events.append(
                        ObservationEvent(
                            run_id=actual_run_id,
                            step=step,
                            observed_ts=ts,
                            source_artifact="transcript.jsonl",
                            event_type="error_repeated",
                            payload={"signature": signature},
                        )
                    )
                seen_error_signatures.add(signature)
            if ENV_BLOCK_RE.search(text):
                events.append(
                    ObservationEvent(
                        run_id=actual_run_id,
                        step=step,
                        observed_ts=ts,
                        source_artifact="transcript.jsonl",
                        event_type="environment_blocked",
                        payload={"summary": str(line.get("summary") or ""), "signature": signature},
                    )
                )

        if kind == "done":
            saw_done = True
            events.append(
                ObservationEvent(
                    run_id=actual_run_id,
                    step=step,
                    observed_ts=ts,
                    source_artifact="transcript.jsonl",
                    event_type="agent_claims_done",
                    payload={"summary": str(line.get("summary") or "")},
                )
            )
            missing_expected = sorted(expected_paths - touched_expected)
            if missing_expected:
                events.append(
                    ObservationEvent(
                        run_id=actual_run_id,
                        step=step,
                        observed_ts=ts,
                        source_artifact="task_tests",
                        event_type="expected_file_missing",
                        payload={
                            "expected_paths": sorted(expected_paths),
                            "missing_paths": missing_expected,
                        },
                    )
                )

    max_step = max((int(line.get("step", 0)) for line in transcript), default=0)
    terminal_step = max_step + 1
    final_success = manifest.get("final_success")
    if final_success is True:
        events.append(
            ObservationEvent(
                run_id=actual_run_id,
                step=terminal_step,
                observed_ts=manifest.get("end_time"),
                source_artifact="run_manifest.json",
                event_type="verifier_pass",
                payload={"termination_reason": manifest.get("termination_reason")},
            )
        )
    elif final_success is False:
        failure_type = _failure_type(verifier_output)
        events.append(
            ObservationEvent(
                run_id=actual_run_id,
                step=terminal_step,
                observed_ts=manifest.get("end_time"),
                source_artifact="run_manifest.json",
                event_type="verifier_fail",
                payload={
                    "termination_reason": manifest.get("termination_reason"),
                    "failure_type": failure_type,
                },
            )
        )
        if saw_done or saw_validation:
            events.append(
                ObservationEvent(
                    run_id=actual_run_id,
                    step=terminal_step,
                    observed_ts=manifest.get("end_time"),
                    source_artifact="run_manifest.json",
                    event_type="verifier_disagreement",
                    payload={
                        "agent_claimed_done": saw_done,
                        "had_validation_attempt": saw_validation,
                        "failure_type": failure_type,
                    },
                )
            )

    return [event.to_dict() for event in sorted(events, key=lambda event: (event.step, event.event_type))]
