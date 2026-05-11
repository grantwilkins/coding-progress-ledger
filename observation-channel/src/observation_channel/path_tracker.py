from __future__ import annotations

import re
import shlex
from typing import Any

from .classify import action_command
from .models import Turn


REDIRECT_RE = re.compile(r"(?:^|[^\d])>>?\s*([^\s;&|]+)")
TEE_RE = re.compile(r"\btee\s+(?:-a\s+)?([^\s;&|]+)")
TOUCH_RE = re.compile(r"\b(?:touch|mkdir|rm|mv|cp|chmod|chown)\s+([^\s;&|]+)")
LINE_RANGE_RE = re.compile(r"^\d+(?::\d+)?$")
DIFF_TARGET_RE = re.compile(r"^\+\+\+\s+(?:[ab]/)?(.+)$", re.MULTILINE)


def first_write_target(turn: Turn) -> str | None:
    args = turn.arguments or {}
    for key in ("path", "file", "filename", "target_path"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _clean(value)

    command = action_command(turn)
    if not command and (turn.tool or "").lower() == "patch":
        command = _patch_text(args)
    if not command:
        return None

    tool = (turn.tool or "").lower()
    if tool == "patch":
        target = _diff_target(command)
        if target:
            return target
    if tool == "create":
        parts = _split(command)
        if len(parts) >= 2:
            return _clean(parts[1])
    if tool in {"edit", "write_file", "edit_file", "create_file"}:
        parts = _split(command)
        if len(parts) >= 2 and not LINE_RANGE_RE.fullmatch(parts[1]):
            return _clean(parts[1])
        return None

    for regex in (REDIRECT_RE, TEE_RE):
        match = regex.search(command)
        if match:
            return _clean(match.group(1))
    return _shell_write_target(command)


def is_source_path(path: str) -> bool:
    normalized = _normalize_workspace_path(path)
    if not normalized:
        return False
    basename = normalized.rsplit("/", 1)[-1]
    root_level = "/" not in normalized
    if basename.startswith("reproduce"):
        return False
    if root_level and basename.startswith("test_"):
        return False
    if root_level and basename.endswith(".py"):
        return False
    return True


def _shell_write_target(command: str) -> str | None:
    parts = _split(command)
    if not parts:
        return None
    cmd = parts[0]
    if cmd == "sed" and any(part == "-i" or part.startswith("-i") or (part.startswith("-") and "i" in part) for part in parts[1:]):
        operands = [part for part in parts[1:] if not part.startswith("-")]
        return _clean(operands[-1]) if len(operands) >= 2 else None
    if cmd not in {"touch", "mkdir", "rm", "mv", "cp", "chmod", "chown"}:
        return None
    operands = [part for part in parts[1:] if not part.startswith("-")]
    if not operands:
        return None
    if cmd in {"mv", "cp"} and len(operands) >= 2:
        return _clean(operands[-1])
    return _clean(operands[0])


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.strip().split()


def _clean(path: Any) -> str:
    return str(path).strip().strip("'\"")


def _patch_text(args: dict[str, Any]) -> str:
    for key in ("patch", "diff", "content"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    return ""


def _diff_target(text: str) -> str | None:
    for match in DIFF_TARGET_RE.finditer(text):
        path = _clean(match.group(1))
        if path != "/dev/null":
            return path
    return None


def _normalize_workspace_path(path: str) -> str:
    normalized = _clean(path)
    for prefix in ("/workspace/repo/", "/workspace/default/", "/workspace/", "/app/", "/repo/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")
