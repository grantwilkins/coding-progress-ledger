from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are solving a coding task in an isolated sandboxed workspace.
Use only the provided tools. Do not assume hidden tests, oracle files, gold
patches, solution files, verifier internals, or final labels. Return exactly
one JSON action per turn. Do not claim done until you have made a reasonable
validation attempt when one is available.

Work as a coding agent: read task.md first, inspect the workspace, search for
relevant files, make targeted edits, and run focused visible checks before
done. If network access or package installation fails, do not retry the same
install loop; look for local dependencies, existing files, or code changes.
For long files, use read_file with start_line and end_line after the default
head+tail view shows omitted lines."""


TOOL_SPECS: list[dict[str, Any]] = [
    {"type": "list_dir", "required": ["path"], "description": "List one workspace directory."},
    {
        "type": "find_files",
        "required": ["pattern"],
        "description": "Find workspace files by name pattern, optionally under path.",
    },
    {
        "type": "grep",
        "required": ["pattern"],
        "description": "Search workspace text with a bounded result set, optionally under path and file_glob.",
    },
    {
        "type": "read_file",
        "required": ["path"],
        "description": "Read one workspace file. Optional start_line/end_line returns a chunk.",
    },
    {"type": "write_file", "required": ["path", "content"], "description": "Write full file contents."},
    {"type": "edit_file", "required": ["path", "instruction"], "description": "Request an edit to a file."},
    {"type": "apply_patch", "required": ["unified_diff"], "description": "Apply a unified diff inside the workspace."},
    {"type": "shell", "required": ["command"], "description": "Run a shell command in the sandbox."},
    {"type": "done", "required": ["summary"], "description": "Stop the agent loop."},
]


def build_prompt(*, task_md: str, transcript_tail: list[dict[str, Any]], max_tail: int = 12) -> str:
    tail = transcript_tail[-max_tail:]
    recent = "\n".join(_summarize(row) for row in tail) or "(no prior actions)"
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "",
            "Response schema:",
            '{"thought_summary": "...", "action": {"type": "read_file", "path": "task.md", "command": null, "content": null, "instruction": null, "summary": null, "start_line": null, "end_line": null, "pattern": null, "file_glob": null, "unified_diff": null}}',
            "",
            "Available action types: list_dir, find_files, grep, read_file, write_file, edit_file, apply_patch, shell, done.",
            "Optional read_file fields: start_line, end_line.",
            "Optional grep fields: path, file_glob.",
            "The controller may reject done if budget_state says more depth or validation is required.",
            "If done is rejected, continue with useful inspection, implementation, or validation.",
            "",
            "Task:",
            task_md,
            "",
            "Recent transcript:",
            recent,
        ]
    )


def _summarize(row: dict[str, Any]) -> str:
    step = row.get("step", "?")
    kind = row.get("kind", "unknown")
    summary = row.get("summary") or row.get("command") or row.get("path") or ""
    pieces = [f"{step}: {kind}: {summary}"]
    if row.get("exit_code") is not None:
        pieces.append(f"exit={row.get('exit_code')}")
    for key in ("stdout_snippet", "stderr_snippet"):
        snippet = str(row.get(key) or "").strip()
        if snippet:
            pieces.append(f"{key}={_compact(snippet)}")
    return " | ".join(pieces)


def _compact(text: str, *, limit: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:240] + " ... " + text[-240:]
