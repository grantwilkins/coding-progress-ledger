from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .action_schema import ModelAction, parse_model_action
from .base import AgentBudget, AgentResult
from .model_client import ModelClient, ScriptedModelClient, model_client_metrics
from .prompt_builder import SYSTEM_PROMPT, TOOL_SPECS, build_prompt
from coding_data_collection.observation import VALIDATION_RE
from coding_data_collection.recording.transcript_recorder import RunRecorder
from coding_data_collection.sandbox.docker_executor import SandboxExecutor, ToolResult


BLOCKED_RE = re.compile(
    r"\b("
    r"blocked|cannot proceed|can't proceed|unable to proceed|no actionable task|"
    r"missing task|missing files?|no such file|command not found|permission denied|"
    r"connection refused|encrypted|decrypt|bad decrypt|not feasible|insufficient information"
    r")\b",
    re.IGNORECASE,
)

NETWORK_BLOCK_RE = re.compile(
    r"\b("
    r"network is unreachable|temporary failure resolving|could not resolve|"
    r"failed to connect|connection timed out|name or service not known|"
    r"no route to host|network.*disabled|connection refused"
    r")\b",
    re.IGNORECASE,
)
DEPENDENCY_BLOCK_RE = re.compile(
    r"\b(no module named|modulenotfounderror|command not found|package .* not found|unable to locate package)\b",
    re.IGNORECASE,
)

TRUNCATION_MARKER = "[cdc:output_truncated]"


class ModelToolLoopBackend:
    name = "model_tool_loop"

    def __init__(
        self,
        model_client: ModelClient,
        *,
        model_name: str = "scripted",
        force_pilot_eligible: bool = False,
        eligible_for_l_gate: bool | None = None,
        pilot_type: str | None = None,
    ) -> None:
        self.model_client = model_client
        self.model_name = model_name
        default_eligible = bool(model_client.provider_backed or force_pilot_eligible)
        self.eligible_for_L_gate = default_eligible if eligible_for_l_gate is None else bool(eligible_for_l_gate)
        self.pilot_type = pilot_type or ("real_agent_pilot" if self.eligible_for_L_gate else "provider_model_smoke" if model_client.provider_backed else "scripted_model_smoke")

    def run(
        self,
        *,
        run_id: str,
        task_md: str,
        workspace_dir: Path,
        sandbox: SandboxExecutor,
        budget: AgentBudget,
        recorder: RunRecorder,
    ) -> AgentResult:
        del run_id, workspace_dir
        transcript = recorder.read_transcript()
        for _ in range(budget.max_steps):
            prompt = build_prompt(task_md=task_md, transcript_tail=transcript)
            action = self._next_valid_action(
                task_prompt=prompt,
                transcript=transcript,
                budget_state={
                    "max_steps": budget.max_steps,
                    "steps_used": recorder.step,
                    "max_wall_time_s": budget.max_wall_time_s,
                    "max_tool_time_s": budget.max_tool_time_s,
                    "min_steps_before_done": budget.min_steps_before_done,
                    "require_validation_before_done": budget.require_validation_before_done,
                    "allow_blocked_done": budget.allow_blocked_done,
                },
                recorder=recorder,
            )
            if action is None:
                return AgentResult(
                    completed=False,
                    termination_reason="invalid_model_action",
                    steps_used=recorder.step,
                    **_usage_kwargs(self.model_client),
                )

            if action.thought_summary:
                recorder.record("thought", summary=action.thought_summary, visible_to_agent=True)
            if action.action_type == "done":
                transcript = recorder.read_transcript()
                allowed, reason = _done_allowed(
                    transcript,
                    summary=action.summary or "",
                    budget=budget,
                    current_step=recorder.step,
                )
                if not allowed:
                    recorder.record(
                        "early_done_denied",
                        summary="done rejected by controller policy",
                        stderr_snippet=reason,
                        visible_to_agent=True,
                    )
                    transcript = recorder.read_transcript()
                    continue
                recorder.record("done", summary=action.summary or "done", visible_to_agent=True)
                return AgentResult(
                    completed=True,
                    termination_reason="agent_done",
                    steps_used=recorder.step,
                    **_usage_kwargs(self.model_client),
                )

            try:
                result = self._execute_action(action, sandbox=sandbox, timeout_s=budget.max_tool_time_s)
            except ValueError as exc:
                recorder.record(
                    "tool_denied",
                    summary="tool action denied",
                    path=action.path,
                    command=action.command,
                    stderr_snippet=str(exc),
                    visible_to_agent=True,
                )
                transcript = recorder.read_transcript()
                continue
            recorder.record(
                action.action_type,
                summary=action.summary or _summary_for_action(action),
                command=action.command,
                path=action.path,
                pattern=action.pattern,
                file_glob=action.file_glob,
                start_line=action.start_line,
                end_line=action.end_line,
                exit_code=result.exit_code,
                stdout_snippet=_bounded_snippet(result.stdout),
                stderr_snippet=_bounded_snippet(result.stderr),
                duration_ms=result.duration_ms,
                visible_to_agent=True,
            )
            for event_kind, message in _classified_tool_events(action, result):
                recorder.record(
                    event_kind,
                    summary=message,
                    command=action.command,
                    path=action.path,
                    stdout_snippet=_bounded_snippet(result.stdout, limit=800),
                    stderr_snippet=_bounded_snippet(result.stderr, limit=800),
                    visible_to_agent=True,
                )
            transcript = recorder.read_transcript()

        recorder.record("done", summary="agent budget exhausted", visible_to_agent=True)
        return AgentResult(
            completed=False,
            termination_reason="budget_exhausted",
            steps_used=recorder.step,
            **_usage_kwargs(self.model_client),
        )

    def _next_valid_action(
        self,
        *,
        task_prompt: str,
        transcript: list[dict[str, Any]],
        budget_state: dict[str, Any],
        recorder: RunRecorder,
        max_retries: int = 2,
    ) -> ModelAction | None:
        for attempt in range(max_retries + 1):
            raw = self.model_client.next_action(
                system_prompt=SYSTEM_PROMPT,
                task_prompt=task_prompt,
                transcript_prefix=transcript,
                tool_specs=TOOL_SPECS,
                budget_state={**budget_state, "retry_attempt": attempt},
            )
            if isinstance(raw, dict) and isinstance(raw.get("provider_adapter_error"), dict):
                error = raw["provider_adapter_error"]
                recorder.record(
                    "provider_adapter_error",
                    summary=str(error.get("message") or "provider adapter error"),
                    stdout_snippet=str(error.get("stdout_snippet") or ""),
                    stderr_snippet=str(error.get("stderr_snippet") or ""),
                    exit_code=error.get("returncode"),
                    visible_to_agent=True,
                )
                transcript = recorder.read_transcript()
                continue
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as exc:
                recorder.record(
                    "model_parse_error",
                    summary="invalid JSON from model",
                    stderr_snippet=str(exc),
                    visible_to_agent=True,
                )
                transcript = recorder.read_transcript()
                continue
            try:
                return parse_model_action(payload)
            except ValueError as exc:
                recorder.record(
                    "invalid_action",
                    summary="invalid model action",
                    stderr_snippet=str(exc),
                    visible_to_agent=True,
                )
                transcript = recorder.read_transcript()
        return None

    def _execute_action(self, action: ModelAction, *, sandbox: SandboxExecutor, timeout_s: int) -> ToolResult:
        if action.action_type == "find_files":
            return sandbox.find_files(action.pattern or "*", path=action.path or ".", timeout_s=timeout_s)
        if action.action_type == "grep":
            return sandbox.grep(
                action.pattern or "",
                path=action.path or ".",
                file_glob=action.file_glob,
                timeout_s=timeout_s,
            )
        if action.action_type == "shell":
            return sandbox.shell(action.command or "", timeout_s=timeout_s)
        if action.action_type == "list_dir":
            return sandbox.list_dir(action.path or ".", timeout_s=timeout_s)
        if action.action_type == "read_file":
            return sandbox.read_file(
                action.path or "",
                timeout_s=timeout_s,
                start_line=action.start_line,
                end_line=action.end_line,
            )
        if action.action_type == "write_file":
            return sandbox.write_file(action.path or "", action.content or "", timeout_s=timeout_s)
        if action.action_type == "edit_file":
            return sandbox.edit_file(action.path or "", action.instruction or "", timeout_s=timeout_s)
        if action.action_type == "apply_patch":
            return sandbox.apply_patch(action.unified_diff or "", timeout_s=timeout_s)
        raise ValueError(f"cannot execute action type: {action.action_type}")


def _summary_for_action(action: ModelAction) -> str:
    if action.action_type == "find_files":
        return f"find_files {action.pattern}"
    if action.action_type == "grep":
        return f"grep {action.pattern}"
    if action.action_type == "shell":
        return "shell tool"
    if action.action_type in {"list_dir", "read_file", "write_file", "edit_file"}:
        return f"{action.action_type} {action.path}"
    if action.action_type == "apply_patch":
        return "apply_patch"
    return action.action_type


def _usage_kwargs(client: ModelClient) -> dict[str, Any]:
    metrics = model_client_metrics(client)
    return {
        "tokens_in": int(metrics.get("total_tokens_in") or 0),
        "tokens_out": int(metrics.get("total_tokens_out") or 0),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
    }


def _done_allowed(
    transcript: list[dict[str, Any]],
    *,
    summary: str,
    budget: AgentBudget,
    current_step: int,
) -> tuple[bool, str]:
    if budget.allow_blocked_done and _blocked_evidence(transcript, summary=summary):
        return True, "blocked evidence present"
    if current_step < budget.min_steps_before_done:
        return (
            False,
            f"done before minimum transcript depth: step {current_step} < {budget.min_steps_before_done}",
        )
    if budget.require_validation_before_done and not _validation_attempt_seen(transcript):
        return False, "done before validation-like shell attempt"
    return True, "done policy satisfied"


def _validation_attempt_seen(transcript: list[dict[str, Any]]) -> bool:
    return any(row.get("kind") == "shell" and VALIDATION_RE.search(_row_text(row)) for row in transcript)


def _blocked_evidence(transcript: list[dict[str, Any]], *, summary: str) -> bool:
    if BLOCKED_RE.search(summary):
        return True
    for row in transcript:
        if row.get("kind") in {"tool_denied", "model_parse_error", "invalid_action", "provider_adapter_error"}:
            return True
        if row.get("kind") == "shell" and row.get("exit_code") not in (None, 0) and BLOCKED_RE.search(_row_text(row)):
            return True
    return False


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("summary", "command", "path", "stdout_snippet", "stderr_snippet", "obs_snippet")
    )


def _bounded_snippet(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 120) // 2)
    omitted = len(text) - (2 * half)
    return (
        text[:half].rstrip()
        + f"\n{TRUNCATION_MARKER} ... omitted {omitted} chars ...\n"
        + text[-half:].lstrip()
    )


def _classified_tool_events(action: ModelAction, result: ToolResult) -> list[tuple[str, str]]:
    text = " ".join([action.command or "", result.stdout, result.stderr])
    events: list[tuple[str, str]] = []
    if result.exit_code not in (None, 0) and action.action_type == "shell":
        if NETWORK_BLOCK_RE.search(text):
            events.append(
                (
                    "network_blocked",
                    "Network is disabled or unreachable in this sandbox. Do not retry package installation; look for an offline/local solution.",
                )
            )
        if DEPENDENCY_BLOCK_RE.search(text):
            events.append(
                (
                    "dependency_missing",
                    "A dependency or command is missing in this sandbox. Prefer existing local dependencies or code changes over repeated installs.",
                )
            )
    if TRUNCATION_MARKER in result.stdout or TRUNCATION_MARKER in result.stderr:
        events.append(("tool_output_truncated", "Tool output was truncated; use chunked reads or narrower searches."))
    if action.action_type == "read_file" and (action.start_line is not None or action.end_line is not None):
        events.append(("chunked_file_read", "Read a bounded file line range."))
    return events
