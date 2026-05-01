from __future__ import annotations

from typing import Any


def infer_events(agent_step: dict[str, Any]) -> list[dict[str, Any]]:
    step = int(agent_step["step"])
    tool = agent_step.get("tool_name") or _first_token(agent_step.get("command") or agent_step.get("action")) or "agent_step"
    evidence = f"step {step}: {tool} action observed"
    return [
        {
            "op": "add",
            "id": f"G{step}",
            "category": "product",
            "description": f"Handle {tool} action at step {step}",
        },
        {"op": "complete", "id": f"G{step}", "evidence": [evidence]},
    ]


def _first_token(value: str | None) -> str | None:
    if not value:
        return None
    return value.split()[0] if value.split() else None

