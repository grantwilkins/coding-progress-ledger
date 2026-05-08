from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_TYPES = {
    "find_files",
    "grep",
    "list_dir",
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "shell",
    "done",
}


@dataclass(frozen=True)
class ModelAction:
    thought_summary: str
    action_type: str
    path: str | None = None
    command: str | None = None
    content: str | None = None
    instruction: str | None = None
    summary: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    pattern: str | None = None
    file_glob: str | None = None
    unified_diff: str | None = None


def parse_model_action(payload: dict[str, Any]) -> ModelAction:
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    thought = str(payload.get("thought_summary") or payload.get("thought") or "").strip()
    action = payload.get("action")
    if not isinstance(action, dict):
        raise ValueError("model response must contain action object")
    action_type = str(action.get("type") or "").strip()
    if action_type not in ACTION_TYPES:
        raise ValueError(f"unsupported action type: {action_type!r}")

    parsed = ModelAction(
        thought_summary=thought,
        action_type=action_type,
        path=_optional_text(action.get("path")),
        command=_optional_text(action.get("command")),
        content=_optional_text(action.get("content")),
        instruction=_optional_text(action.get("instruction")),
        summary=_optional_text(action.get("summary")),
        start_line=_optional_int(action.get("start_line")),
        end_line=_optional_int(action.get("end_line")),
        pattern=_optional_text(action.get("pattern")),
        file_glob=_optional_text(action.get("file_glob")),
        unified_diff=_optional_text(action.get("unified_diff") or action.get("diff")),
    )
    _validate_required_fields(parsed)
    return parsed


def _validate_required_fields(action: ModelAction) -> None:
    if action.action_type == "shell" and not action.command:
        raise ValueError("shell action requires command")
    if action.action_type == "find_files" and not action.pattern:
        raise ValueError("find_files action requires pattern")
    if action.action_type == "grep" and not action.pattern:
        raise ValueError("grep action requires pattern")
    if action.action_type in {"list_dir", "read_file"} and not action.path:
        raise ValueError(f"{action.action_type} action requires path")
    if action.action_type == "read_file":
        if action.start_line is not None and action.start_line < 1:
            raise ValueError("read_file start_line must be >= 1")
        if action.end_line is not None and action.end_line < 1:
            raise ValueError("read_file end_line must be >= 1")
        if action.start_line is not None and action.end_line is not None and action.start_line > action.end_line:
            raise ValueError("read_file start_line must be <= end_line")
    if action.action_type == "write_file" and (not action.path or action.content is None):
        raise ValueError("write_file action requires path and content")
    if action.action_type == "edit_file" and not action.path:
        raise ValueError("edit_file action requires path")
    if action.action_type == "apply_patch" and not action.unified_diff:
        raise ValueError("apply_patch action requires unified_diff")
    if action.action_type == "done" and not action.summary:
        raise ValueError("done action requires summary")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("line numbers must be integers")
    return int(value)
