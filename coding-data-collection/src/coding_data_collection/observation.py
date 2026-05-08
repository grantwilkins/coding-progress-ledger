from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import utc_now
from .protocol import VERSIONS

VALIDATION_RE = re.compile(
    r"("
    r"\b(pytest|unittest|lint|mypy|ruff)\b|"
    r"\b(cargo|go|npm|make)\s+test\b|"
    r"\brun-tests\.sh\b|"
    r"\bpy_compile\b|"
    r"\bfinal[-_]check\b|"
    r"\bexample matches:\s*true\b|"
    r"\brunning basic .{0,80}\btest\b|"
    r"\b(completed|processed) successfully\b|"
    r"validation|validate|"
    r"\b(cmp|diff)\b|"
    r"\b(pip|pip3)\s+--version\b|"
    r"\bpython[0-9.]*\s+-m\s+pip\s+--version\b|"
    r"\b(pip|pip3)\b.*(?:^|\s)--dry-run\b"
    r")",
    re.IGNORECASE,
)
FAIL_RE = re.compile(
    r"\b(fail|failed|error|traceback|assertionerror|exception|no such file|command not found)\b",
    re.IGNORECASE,
)
ENV_BLOCK_RE = re.compile(
    r"\b(no module named|modulenotfounderror|command not found|permission denied|connection refused)\b",
    re.IGNORECASE,
)
NETWORK_BLOCK_RE = re.compile(
    r"\b(network is unreachable|temporary failure resolving|could not resolve|failed to connect|connection timed out|network.*disabled)\b",
    re.IGNORECASE,
)
DEPENDENCY_BLOCK_RE = re.compile(
    r"\b(no module named|modulenotfounderror|command not found|package .* not found|unable to locate package)\b",
    re.IGNORECASE,
)
ORACLE_RE = re.compile(
    r"\b(solution\.sh|solution_reference|oracle|gold patch|gold_patch|test_patch|"
    r"tests?/|test_outputs\.py|verifier\.sh|verifier_tests)\b",
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
            "schema_version": VERSIONS.observation_event_schema_version,
            "run_id": self.run_id,
            "step": self.step,
            "observed_ts": self.observed_ts,
            "source_artifact": self.source_artifact,
            "event_type": self.event_type,
            "payload": self.payload,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def build_observation_events(
    transcript: list[dict[str, Any]],
    *,
    run_id: str,
    verifier_exit_code: int | None = None,
    expected_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    expected_paths = expected_paths or set()
    out: list[ObservationEvent] = []
    signatures: Counter[str] = Counter()
    max_step = max((int(row.get("step", 0)) for row in transcript), default=0)
    done_seen = False
    written_paths: set[str] = set()
    read_paths: Counter[str] = Counter()

    for row in transcript:
        step = int(row.get("step", 0))
        kind = str(row.get("kind") or "")
        text = _text(row)
        ts = row.get("ts") or row.get("timestamp") or utc_now()

        if kind == "shell":
            visible_to_agent = _visible_to_agent(row)
            source_artifact = _shell_source_artifact(row)
            payload = {
                "command": row.get("command"),
                "exit_code": row.get("exit_code"),
                "stdout_snippet": row.get("stdout_snippet") or row.get("obs_snippet"),
                "stderr_snippet": row.get("stderr_snippet"),
                "visible_to_agent": visible_to_agent,
            }
            if VALIDATION_RE.search(text):
                out.append(_event(run_id, step, ts, "validation_attempt", payload, source_artifact=source_artifact))
                if row.get("exit_code") == 0:
                    out.append(_event(run_id, step, ts, "validation_pass_observed", payload, source_artifact=source_artifact))
                elif row.get("exit_code") not in (None, 0) or FAIL_RE.search(text):
                    out.append(_event(run_id, step, ts, "validation_fail_observed", payload, source_artifact=source_artifact))
            if row.get("exit_code") not in (None, 0) or (row.get("exit_code") is None and FAIL_RE.search(text)):
                sig = _signature(text)
                err_payload = {**payload, "normalized_error_signature": sig}
                out.append(_event(run_id, step, ts, "error_observed", err_payload, source_artifact=source_artifact))
                if sig:
                    signatures[sig] += 1
                    if signatures[sig] > 1:
                        out.append(_event(run_id, step, ts, "error_repeated", err_payload, source_artifact=source_artifact))
            if ENV_BLOCK_RE.search(text):
                out.append(_event(run_id, step, ts, "environment_blocked", payload, source_artifact=source_artifact))
            if NETWORK_BLOCK_RE.search(text):
                out.append(_event(run_id, step, ts, "network_blocked", payload, source_artifact=source_artifact))
            if DEPENDENCY_BLOCK_RE.search(text):
                out.append(_event(run_id, step, ts, "dependency_missing", payload, source_artifact=source_artifact))

        if kind in {"write_file", "edit_file", "patch", "apply_patch"}:
            path = _normalize_path(row.get("path")) or _patch_output_path(text)
            if path:
                written_paths.add(path)
            out.append(
                _event(
                    run_id,
                    step,
                    ts,
                    "product_file_edited" if kind in {"edit_file", "apply_patch", "patch"} else "product_file_written",
                    {"path": path, "visible_to_agent": True},
                )
            )

        if kind in {"read_file", "open_file"} and ORACLE_RE.search(text):
            out.append(
                _event(
                    run_id,
                    step,
                    ts,
                    "oracle_artifact_read",
                    {"path": _normalize_path(row.get("path")), "visible_to_agent": True},
                )
            )

        if kind == "read_file":
            path = _normalize_path(row.get("path"))
            if path:
                read_paths[path] += 1
                if row.get("start_line") or row.get("end_line"):
                    out.append(
                        _event(
                            run_id,
                            step,
                            ts,
                            "chunked_file_read",
                            {"path": path, "visible_to_agent": True},
                        )
                    )
                if read_paths[path] > 1:
                    out.append(
                        _event(
                            run_id,
                            step,
                            ts,
                            "repeated_file_inspection",
                            {"path": path, "count": read_paths[path], "visible_to_agent": True},
                        )
                    )
            if "[cdc:output_truncated]" in text:
                out.append(
                    _event(
                        run_id,
                        step,
                        ts,
                        "tool_output_truncated",
                        {"path": path, "visible_to_agent": True},
                    )
                )

        if kind in {
            "model_parse_error",
            "invalid_action",
            "tool_denied",
            "early_done_denied",
            "network_blocked",
            "dependency_missing",
            "tool_output_truncated",
            "repeated_file_inspection",
            "chunked_file_read",
        }:
            out.append(
                _event(
                    run_id,
                    step,
                    ts,
                    kind,
                    {
                        "summary": row.get("summary"),
                        "path": _normalize_path(row.get("path")),
                        "command": row.get("command"),
                        "stderr_snippet": row.get("stderr_snippet"),
                        "visible_to_agent": True,
                    },
                )
            )

        if kind == "done":
            done_seen = True
            out.append(_event(run_id, step, ts, "agent_claims_done", {"visible_to_agent": True}))

    terminal_step = max_step + 1
    if verifier_exit_code is not None:
        verifier_type = "verifier_pass" if verifier_exit_code == 0 else "verifier_fail"
        out.append(
            _event(
                run_id,
                terminal_step,
                utc_now(),
                verifier_type,
                {"exit_code": verifier_exit_code, "visible_to_agent": False},
                source_artifact="verifier_output.txt",
            )
        )
        if done_seen and verifier_exit_code != 0:
            out.append(
                _event(
                    run_id,
                    terminal_step,
                    utc_now(),
                    "verifier_disagreement",
                    {"reason": "agent_claimed_done_then_verifier_failed", "visible_to_agent": False},
                    source_artifact="run_manifest.json",
                )
            )

    normalized_expected_paths = {_normalize_path(path) for path in expected_paths}
    missing_expected_paths = normalized_expected_paths - written_paths
    if done_seen and missing_expected_paths:
        out.append(
            _event(
                run_id,
                terminal_step,
                utc_now(),
                "expected_file_missing",
                {
                    "expected_paths": sorted(normalized_expected_paths),
                    "missing_paths": sorted(missing_expected_paths),
                    "written_paths": sorted(written_paths),
                    "visible_to_agent": False,
                },
                source_artifact="task_metadata.json",
            )
        )

    return [event.to_dict() for event in out]


def _event(
    run_id: str,
    step: int,
    ts: str | None,
    event_type: str,
    payload: dict[str, Any],
    *,
    source_artifact: str = "transcript.jsonl",
) -> ObservationEvent:
    return ObservationEvent(
        run_id=run_id,
        step=step,
        observed_ts=ts,
        source_artifact=source_artifact,
        event_type=event_type,
        payload=payload,
    )


def _text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("summary", "command", "path", "obs_snippet", "stdout_snippet", "stderr_snippet")
    )


def _visible_to_agent(row: dict[str, Any]) -> bool:
    if "visible_to_agent" in row:
        return bool(row["visible_to_agent"])
    return row.get("summary") not in {"oracle_phase", "verifier_phase"}


def _shell_source_artifact(row: dict[str, Any]) -> str:
    if row.get("summary") == "verifier_phase":
        return "verifier_output.txt"
    if row.get("summary") == "oracle_phase":
        return "oracle_workspace_snapshot"
    return "transcript.jsonl"


def _signature(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return normalized[:160] or None


def _normalize_path(path: Any) -> str:
    if path is None:
        return ""
    text = str(path).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return re.sub(r"/+", "/", text)


def _patch_output_path(text: str) -> str:
    match = re.search(r"\bpatching file ([^\s]+)", text)
    if not match:
        return ""
    return _normalize_path(match.group(1))
