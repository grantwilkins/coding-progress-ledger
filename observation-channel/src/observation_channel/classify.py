from __future__ import annotations

import json
import re
import shlex
from typing import Any

from .models import Category, Turn


SWE_AGENT_TOOLS: dict[str, Category] = {
    "open": Category.INVESTIGATION,
    "goto": Category.INVESTIGATION,
    "scroll_up": Category.INVESTIGATION,
    "scroll_down": Category.INVESTIGATION,
    "search_file": Category.INVESTIGATION,
    "search_dir": Category.INVESTIGATION,
    "find_file": Category.INVESTIGATION,
    "ls": Category.INVESTIGATION,
    "create": Category.PRODUCT,
    "edit": Category.PRODUCT,
    "submit": Category.ARTIFACT,
}

HERMES_TOOLS: dict[str, Category] = {
    "read_file": Category.INVESTIGATION,
    "list_directory": Category.INVESTIGATION,
    "find_file": Category.INVESTIGATION,
    "grep": Category.INVESTIGATION,
    "search": Category.INVESTIGATION,
    "search_files": Category.INVESTIGATION,
    "web_search": Category.INVESTIGATION,
    "fetch_url": Category.INVESTIGATION,
    "browse": Category.INVESTIGATION,
    "write_file": Category.PRODUCT,
    "patch": Category.PRODUCT,
    "edit_file": Category.PRODUCT,
    "create_file": Category.PRODUCT,
    "run_tests": Category.VALIDATION,
    "pytest": Category.VALIDATION,
    "pip": Category.ENVIRONMENT,
    "pip_install": Category.ENVIRONMENT,
    "apt_get": Category.ENVIRONMENT,
}

ARTIFACT_TOOLS = {
    "submit",
    "submit_answer",
    "final_response",
    "task_complete",
    "finish",
    "done",
}

SHELL_PASSTHROUGH_TOOLS = {"bash", "shell", "terminal", "run", "execute", "execute_code"}

INSTALL_RE = re.compile(
    r"\b("
    r"pip\s+install|pip3\s+install|python[0-9.]*\s+-m\s+pip\s+install|"
    r"uv\s+(add|sync|pip\s+install)|"
    r"apt-get\s+install|apt\s+install|conda\s+install|mamba\s+install|"
    r"npm\s+install|pnpm\s+install|yarn\s+install|cargo\s+install"
    r")\b",
    re.IGNORECASE,
)

TEST_RE = re.compile(
    r"\b("
    r"pytest|unittest|nosetests|tox|jest|go\s+test|cargo\s+test|npm\s+test|"
    r"pnpm\s+test|yarn\s+test|rspec|make\s+test|ctest"
    r")\b",
    re.IGNORECASE,
)

STATE_CHANGE_RE = re.compile(
    r"("
    r"<<\s*['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?|"
    r">>?\s*\S+|"
    r"\btee\s+(-a\s+)?\S+|"
    r"\bsed\s+(-[A-Za-z]*i[A-Za-z]*|-i(?:\s|$))|"
    r"\b(rm|mv|cp|touch|mkdir|chmod|chown)\b|"
    r"\bpython[0-9.]*\s+-c\s+.*\b(open|write_text|Path\().*"
    r")",
    re.IGNORECASE | re.DOTALL,
)

SCRIPT_RE = re.compile(
    r"^\s*("
    r"cd\s+\S+\s*&&\s*"
    r")?("
    r"python[0-9.]*\b|"
    r"bash\b|sh\b|zsh\b|"
    r"\./[\w./-]+|"
    r"make\b|node\b|ruby\b|perl\b|Rscript\b|R\b"
    r")",
    re.IGNORECASE,
)

READ_ONLY_RE = re.compile(
    r"^\s*("
    r"ls|cat|head|tail|pwd|grep|rg|find|which|wc|tree|stat|file|less|more|"
    r"sed\s+-n|awk|git\s+(status|diff|show|log|ls-files|grep)"
    r")\b",
    re.IGNORECASE,
)


def classify_turn(turn: Turn) -> Category | None:
    if turn.kind != "action":
        return None
    tool = (turn.tool or "").strip()
    tool_l = tool.lower()
    command = action_command(turn)
    first = first_token(command).lower()

    if tool_l in ARTIFACT_TOOLS or first in ARTIFACT_TOOLS:
        return Category.ARTIFACT

    if tool_l in SWE_AGENT_TOOLS:
        return SWE_AGENT_TOOLS[tool_l]
    if tool_l in HERMES_TOOLS:
        return HERMES_TOOLS[tool_l]

    if tool_l and tool_l not in SHELL_PASSTHROUGH_TOOLS:
        return classify_bash(command) if command else Category.INVESTIGATION
    return classify_bash(command)


def classify_bash(command: str) -> Category:
    if not command:
        return Category.INVESTIGATION
    if _has_artifact_marker(command):
        return Category.ARTIFACT
    if INSTALL_RE.search(command):
        return Category.ENVIRONMENT
    if TEST_RE.search(command):
        return Category.VALIDATION
    if STATE_CHANGE_RE.search(command):
        return Category.PRODUCT
    if SCRIPT_RE.search(command):
        return Category.VALIDATION
    if READ_ONLY_RE.search(command):
        return Category.INVESTIGATION
    return Category.INVESTIGATION


def action_command(turn: Turn) -> str:
    if turn.command:
        return turn.command
    args = turn.arguments or {}
    for key in ("command", "cmd", "script", "code"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return ""


def first_token(text: str) -> str:
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        parts = text.strip().split()
    return parts[0] if parts else ""


def coerce_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"command": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return None


def _has_artifact_marker(command: str) -> bool:
    stripped = command.strip().lower()
    return any(stripped == marker or stripped.startswith(f"{marker} ") for marker in ARTIFACT_TOOLS)
