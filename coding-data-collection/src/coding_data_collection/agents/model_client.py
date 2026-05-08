from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelClientConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens_out: int = 2048
    timeout_s: int = 120


class ModelClient(Protocol):
    config: ModelClientConfig
    provider_backed: bool

    def next_action(
        self,
        *,
        system_prompt: str,
        task_prompt: str,
        transcript_prefix: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        budget_state: dict[str, Any],
    ) -> str | dict[str, Any]:
        ...

    def metrics(self) -> dict[str, Any]:
        ...


class ScriptedModelClient:
    provider_backed = False

    def __init__(self, actions: list[str | dict[str, Any]], *, model_name: str = "scripted") -> None:
        self._actions = list(actions)
        self._index = 0
        self.config = ModelClientConfig(provider="scripted", model_name=model_name)
        self._calls = 0

    @classmethod
    def from_jsonl(cls, path: Path, *, model_name: str = "scripted") -> "ScriptedModelClient":
        actions: list[str | dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                actions.append(json.loads(line))
            except json.JSONDecodeError:
                actions.append(line)
        return cls(actions, model_name=model_name)

    def next_action(
        self,
        *,
        system_prompt: str,
        task_prompt: str,
        transcript_prefix: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        budget_state: dict[str, Any],
    ) -> str | dict[str, Any]:
        del system_prompt, task_prompt, transcript_prefix, tool_specs, budget_state
        self._calls += 1
        if self._index >= len(self._actions):
            return {
                "thought_summary": "No scripted actions remain.",
                "action": {"type": "done", "summary": "budget exhausted"},
            }
        payload = self._actions[self._index]
        self._index += 1
        return payload

    def metrics(self) -> dict[str, Any]:
        return {
            "total_model_calls": self._calls,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "estimated_cost_usd": None,
        }


class ProviderModelClient:
    """Provider-neutral client implemented by an external command adapter.

    The command receives a JSON request on stdin and must write either a model
    action JSON object or an envelope with `action` and optional `usage` fields
    to stdout.
    """

    provider_backed = True

    def __init__(self, *, command: str | None = None, config: ModelClientConfig) -> None:
        command = command or os.environ.get("CDC_MODEL_CLIENT_COMMAND")
        if not command:
            raise ValueError("ProviderModelClient requires --model-command or CDC_MODEL_CLIENT_COMMAND")
        self.command = shlex.split(command)
        self.config = config
        self._calls = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._estimated_cost_usd = 0.0
        self._cost_seen = False
        self._cost_credits = 0.0
        self._cost_credits_seen = False
        self._provider_calls: list[dict[str, Any]] = []

    def next_action(
        self,
        *,
        system_prompt: str,
        task_prompt: str,
        transcript_prefix: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        budget_state: dict[str, Any],
    ) -> str | dict[str, Any]:
        request = {
            "system_prompt": system_prompt,
            "task_prompt": task_prompt,
            "transcript_prefix": transcript_prefix,
            "tool_specs": tool_specs,
            "budget_state": budget_state,
            "model": {
                "provider": self.config.provider,
                "name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens_out": self.config.max_tokens_out,
            },
        }
        try:
            proc = subprocess.run(
                self.command,
                input=json.dumps(request, sort_keys=True),
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            self._calls += 1
            stderr = exc.stderr or ""
            return _provider_adapter_error(
                "provider_adapter_timeout",
                f"provider adapter timed out after {self.config.timeout_s}s",
                stderr_snippet=str(stderr)[-2000:],
            )
        self._calls += 1
        if proc.returncode != 0:
            return _provider_adapter_error(
                "provider_adapter_exit",
                f"provider adapter exited {proc.returncode}",
                returncode=proc.returncode,
                stdout_snippet=proc.stdout[-2000:],
                stderr_snippet=proc.stderr[-2000:],
            )
        return self._parse_provider_stdout(proc.stdout)

    def metrics(self) -> dict[str, Any]:
        return {
            "total_model_calls": self._calls,
            "total_tokens_in": self._tokens_in,
            "total_tokens_out": self._tokens_out,
            "estimated_cost_usd": self._estimated_cost_usd if self._cost_seen else None,
            "total_cost_credits": self._cost_credits if self._cost_credits_seen else None,
            "provider_calls": list(self._provider_calls),
            "last_provider_call": self._provider_calls[-1] if self._provider_calls else None,
            "fallback_call_count": sum(bool(call.get("fallback_used")) for call in self._provider_calls),
            "resolved_models": sorted(
                {
                    str(call.get("resolved_model"))
                    for call in self._provider_calls
                    if call.get("resolved_model") is not None
                }
            ),
        }

    def _parse_provider_stdout(self, text: str) -> str | dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict) and "usage" in payload:
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            self._tokens_in += int(usage.get("tokens_in") or usage.get("input_tokens") or 0)
            self._tokens_out += int(usage.get("tokens_out") or usage.get("output_tokens") or 0)
            if usage.get("estimated_cost_usd") is not None:
                self._estimated_cost_usd += float(usage["estimated_cost_usd"])
                self._cost_seen = True
            if usage.get("cost_credits") is not None:
                self._cost_credits += float(usage["cost_credits"])
                self._cost_credits_seen = True
            provider = payload.get("provider")
            if isinstance(provider, dict):
                self._provider_calls.append(provider)
            action = payload.get("action")
            if isinstance(action, dict):
                return action
        return payload


def model_client_metrics(client: ModelClient) -> dict[str, Any]:
    try:
        return client.metrics()
    except AttributeError:
        return {
            "total_model_calls": None,
            "total_tokens_in": None,
            "total_tokens_out": None,
            "estimated_cost_usd": None,
        }


def _provider_adapter_error(
    kind: str,
    message: str,
    *,
    returncode: int | None = None,
    stdout_snippet: str = "",
    stderr_snippet: str = "",
) -> dict[str, Any]:
    return {
        "provider_adapter_error": {
            "kind": kind,
            "message": message,
            "returncode": returncode,
            "stdout_snippet": stdout_snippet,
            "stderr_snippet": stderr_snippet,
        }
    }
