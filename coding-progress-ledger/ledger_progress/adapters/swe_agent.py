from __future__ import annotations

from typing import Any


INVESTIGATION = {"find_file", "search_dir", "search_file", "grep", "ls", "open", "goto", "scroll_up", "scroll_down"}
ENVIRONMENT = {"pip", "pip3", "apt-get", "apt", "conda", "uv"}


def infer_events(agent_step: dict[str, Any]) -> list[dict[str, Any]]:
    step = int(agent_step["step"])
    command = (agent_step.get("command") or agent_step.get("action") or "").strip()
    tool = agent_step.get("tool_name") or _first_token(command)
    if not tool:
        return []

    category = _category(tool, command, agent_step.get("files_touched") or [])
    subtask_id = f"SW{step}"
    ops = [
        {
            "op": "add",
            "id": subtask_id,
            "category": category,
            "description": _description(category, tool, command, step),
        }
    ]
    status_op = "complete" if agent_step.get("observation") or tool == "submit" else "start"
    ops.append({"op": status_op, "id": subtask_id, "evidence": [_evidence(step, tool, command, agent_step.get("observation"))]})
    return ops


def _category(tool: str, command: str, files_touched: list[str]) -> str:
    lower_tool = tool.lower()
    lower_command = command.lower()
    if lower_tool in INVESTIGATION:
        return "investigation"
    if lower_tool in {"edit", "create"}:
        if any(path.startswith("docs/") or path.endswith(".md") for path in files_touched) or ".md" in lower_command:
            return "documentation"
        return "product"
    if lower_tool in {"pytest", "tox", "unittest"} or lower_command.startswith("python "):
        return "validation"
    if lower_tool in ENVIRONMENT or " install " in f" {lower_command} ":
        return "environment"
    if lower_tool == "submit":
        return "artifact"
    return "product"


def _description(category: str, tool: str, command: str, step: int) -> str:
    target = _short(command or tool)
    prefixes = {
        "investigation": "Inspect code or repo state",
        "product": "Change implementation",
        "validation": "Run validation",
        "environment": "Adjust execution environment",
        "artifact": "Submit final artifact",
        "documentation": "Change documentation",
    }
    return f"{prefixes[category]} via {target} at step {step}"


def _evidence(step: int, tool: str, command: str, observation: str | None) -> str:
    base = f"step {step}: {tool}"
    if command:
        base = f"{base} command `{_short(command)}`"
    if observation:
        return f"{base}; observation `{_short(observation)}`"
    return f"{base} issued"


def _first_token(value: str | None) -> str | None:
    if not value:
        return None
    return value.split()[0] if value.split() else None


def _short(value: str, limit: int = 120) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

