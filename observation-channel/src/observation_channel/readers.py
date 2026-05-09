from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from .classify import coerce_arguments, first_token
from .models import Turn


BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
RETURNCODE_RE = re.compile(r"<returncode>\s*(.*?)\s*</returncode>", re.DOTALL | re.IGNORECASE)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\$[0-9A-Fa-f]+")
GENERIC_SHELL_TOOLS = {"bash", "shell", "terminal", "bash_command"}


def swe_agent_turns(row: dict[str, Any], *, source: str = "swe-agent") -> list[Turn]:
    turns: list[Turn] = []
    context = _context(row, source)
    for item in _list_value(row.get("trajectory"), "trajectory"):
        role = _role(item, "role")
        if role == "system":
            _append(turns, context, "system", response=_first_str(item, "system_prompt", "content", "message", "text"))
        elif role in {"user", "human"}:
            text = _first_str(item, "observation", "content", "message", "text")
            kind = "observation" if turns and turns[-1].kind == "action" else "user"
            _append(turns, context, kind, response=text)
        elif role in {"assistant", "ai", "gpt"}:
            command = _command_from_text(_first_str(item, "action", "command", "content", "message", "text"))
            _append(
                turns,
                context,
                "action",
                tool=_first_str(item, "tool_name", "tool", "name") or first_token(command),
                command=command,
                arguments=coerce_arguments(item.get("arguments") or item.get("args")),
            )
    return turns


def mini_swe_turns(row: dict[str, Any], *, source: str = "terminalbench") -> list[Turn]:
    turns: list[Turn] = []
    context = _context(row, source)
    for item in _messages_from_row(row):
        role = _role(item, "role", "from", "src")
        content = _first_str(item, "content", "value", "message", "text", "msg")
        obs = _first_str(item, "obs", "observation", "output")

        if role == "system":
            _append_if_text(turns, context, "system", content)
        elif role in {"user", "human"}:
            kind = "observation" if _has_return_tags(content) else "user"
            _append_if_text(turns, context, kind, extract_tagged_output(content) or content)
        elif role in {"assistant", "gpt", "agent", ""}:
            command = _command_from_tools(item.get("tools")) or extract_bash_command(content) or _first_str(item, "command", "action")
            _append_action_if_command(turns, context, command)
        elif role == "tool":
            _append_if_text(turns, context, "observation", extract_tagged_output(content) or content)
        else:
            _append_if_text(turns, context, "user", content)

        _append_if_text(turns, context, "observation", extract_tagged_output(obs) or obs)
    return turns


def hermes_turns(row: dict[str, Any], *, source: str = "hermes") -> list[Turn]:
    turns: list[Turn] = []
    context = _context(row, source)
    conversations = _list_value(row.get("conversations") or row.get("messages"), "conversations/messages")
    for item in conversations:
        role = _role(item, "from", "role")
        content = _first_str(item, "value", "content", "message", "text")
        if role == "system":
            _append(turns, context, "system", response=content)
        elif role in {"human", "user"}:
            _append(turns, context, "user", response=content)
        elif role in {"gpt", "assistant"}:
            for tool_call in extract_tool_calls(content):
                args = tool_call.get("arguments")
                args = args if isinstance(args, dict) else None
                command = _command_from_args(args)
                tool = str(tool_call.get("name") or "")
                if tool.lower() in GENERIC_SHELL_TOOLS and command:
                    tool = first_token(command)
                _append(turns, context, "action", tool=tool, command=command, arguments=args)
        elif role == "tool":
            for response in extract_tool_responses(content) or [content]:
                _append(turns, context, "observation", response=response)
    return turns


def rows_to_turns(rows: Iterable[dict[str, Any]], *, source: str) -> Iterator[tuple[str, list[Turn]]]:
    readers = {"swe-agent": swe_agent_turns, "hermes": hermes_turns, "terminalbench": mini_swe_turns}
    if source not in readers:
        raise ValueError(f"unsupported source: {source}")
    for index, row in enumerate(rows):
        instance_id = _instance_id(row) or f"{source}-{index:06d}"
        turns = readers[source](row, source=source)
        if not turns:
            raise ValueError(f"{source}:{instance_id}: no canonical turns extracted")
        yield instance_id, turns


def extract_bash_command(content: str) -> str | None:
    match = BASH_BLOCK_RE.search(content or "")
    return match.group(1).strip() if match else None


def extract_tagged_output(content: str) -> str | None:
    parts = []
    if ret := RETURNCODE_RE.search(content or ""):
        parts.append(f"returncode={ret.group(1).strip()}")
    if out := OUTPUT_RE.search(content or ""):
        parts.append(out.group(1).strip())
    return "\n".join(part for part in parts if part) or None


def extract_tool_calls(content: str) -> list[dict[str, Any]]:
    return [_json_object(match.group(1), "tool_call") for match in TOOL_CALL_RE.finditer(content or "")]


def extract_tool_responses(content: str) -> list[str]:
    responses = []
    for match in TOOL_RESPONSE_RE.finditer(content or ""):
        raw = match.group(1).strip()
        obj = _json_object(raw, "tool_response") if raw.startswith("{") else None
        responses.append(_first_str(obj, "content") if obj else raw)
    return responses


def _append(
    turns: list[Turn],
    context: dict[str, str],
    kind: str,
    *,
    response: str | None = None,
    tool: str | None = None,
    command: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    if kind == "action" and not (tool or command):
        raise ValueError(f"{context['source']}:{context['instance_id']}: action without tool or command")
    turns.append(
        Turn(
            step=len(turns) + 1,
            kind=kind,
            tool=tool or None,
            command=command or None,
            response=response or None,
            arguments=arguments,
            source=context["source"],
            instance_id=context["instance_id"],
            metadata={"exit_status": context["exit_status"]},
        )
    )


def _append_if_text(turns: list[Turn], context: dict[str, str], kind: str, text: str | None) -> None:
    if text and not PLACEHOLDER_RE.fullmatch(text.strip()):
        _append(turns, context, kind, response=text)


def _append_action_if_command(turns: list[Turn], context: dict[str, str], command: str | None) -> None:
    if command and not PLACEHOLDER_RE.fullmatch(command.strip()):
        _append(turns, context, "action", tool=first_token(command) or "bash", command=command)


def _messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    if messages := _maybe_list(row.get("messages"), "messages"):
        return messages
    raw = _maybe_dict(row.get("trajectory_raw"), "trajectory_raw")
    if raw and (messages := _maybe_list(raw.get("messages"), "trajectory_raw.messages")):
        return messages
    steps = _list_value(row.get("steps"), "steps")
    return [msg for step in steps for msg in _step_messages(step)]


def _step_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    messages = step.get("messages")
    if isinstance(messages, list):
        return [msg for msg in messages if isinstance(msg, dict)]
    return [step]


def _context(row: dict[str, Any], source: str) -> dict[str, str]:
    return {"source": source, "instance_id": _instance_id(row), "exit_status": _exit_status(row)}


def _command_from_text(text: str) -> str:
    return extract_bash_command(_strip_thoughts(text)) or _strip_thoughts(text)


def _command_from_tools(tools: Any) -> str | None:
    if not isinstance(tools, list):
        return None
    for tool in tools:
        if isinstance(tool, dict) and (command := _command_from_args(tool)):
            return command
    return None


def _command_from_args(args: dict[str, Any] | None) -> str | None:
    if not args:
        return None
    for key in ("cmd", "command", "script", "code"):
        if isinstance(args.get(key), str) and args[key].strip():
            return args[key]
    return None


def _list_value(value: Any, field: str) -> list[dict[str, Any]]:
    parsed = _json_value(value, field)
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise ValueError(f"{field} must be a list of objects")
    return parsed


def _maybe_list(value: Any, field: str) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    return _list_value(value, field)


def _maybe_dict(value: Any, field: str) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    parsed = _json_value(value, field)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be an object")
    return parsed


def _json_value(value: Any, field: str) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} contains invalid JSON") from exc
    raise ValueError(f"{field} is missing")


def _json_object(value: str, field: str) -> dict[str, Any]:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} contains invalid JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{field} must be a JSON object")
    return obj


def _first_str(obj: dict[str, Any] | None, *keys: str) -> str:
    if not obj:
        return ""
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return ""


def _role(obj: dict[str, Any], *keys: str) -> str:
    return _first_str(obj, *keys).lower()


def _instance_id(row: dict[str, Any]) -> str:
    for key in ("instance_id", "trial_id", "traj_id", "id", "seed_id", "task_name", "task_slug"):
        if str(row.get(key) or "").strip():
            return str(row[key])
    return ""


def _exit_status(row: dict[str, Any]) -> str:
    if row.get("exit_status") is not None:
        return str(row["exit_status"])
    if row.get("resolved") is not None:
        return "resolved" if bool(row["resolved"]) else "unresolved"
    if row.get("reward") is not None:
        return f"reward={row['reward']}"
    return "unknown"


def _has_return_tags(content: str) -> bool:
    lowered = (content or "").lower()
    return "<returncode>" in lowered or "<output>" in lowered


def _strip_thoughts(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()
