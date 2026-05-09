from __future__ import annotations

import re
import shlex
from typing import Any

from .classify import action_command
from .models import Turn


REDIRECT_RE = re.compile(r"(?:^|[^\d])>>?\s*([^\s;&|]+)")
TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s;&|]+)")
TOUCH_RE = re.compile(r"\b(?:touch|mkdir|rm|mv|cp|chmod|chown)\s+([^\s;&|]+)")


def first_write_target(turn: Turn) -> str | None:
    args = turn.arguments or {}
    for key in ("path", "file", "filename", "target_path"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)

    command = action_command(turn)
    if not command:
        return None

    tool = (turn.tool or "").lower()
    if tool == "create":
        parts = _split(command)
        if len(parts) >= 2:
            return _clean(parts[1])
    if tool in {"edit", "write_file", "edit_file", "create_file"}:
        parts = _split(command)
        if len(parts) >= 2:
            return _clean(parts[1])

    for regex in (REDIRECT_RE, TEE_RE, TOUCH_RE):
        match = regex.search(command)
        if match:
            return _clean(match.group(1))
    return None


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.strip().split()


def _clean(path: Any) -> str:
    return str(path).strip().strip("'\"")
