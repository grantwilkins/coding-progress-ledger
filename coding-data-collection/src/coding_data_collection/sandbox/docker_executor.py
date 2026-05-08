from __future__ import annotations

import subprocess
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .path_guard import PathGuard


@dataclass(frozen=True)
class ToolResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0


class SandboxExecutor(Protocol):
    def shell(self, command: str, *, timeout_s: int) -> ToolResult:
        ...

    def list_dir(self, path: str, *, timeout_s: int) -> ToolResult:
        ...

    def read_file(
        self,
        path: str,
        *,
        timeout_s: int,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        ...

    def write_file(self, path: str, content: str, *, timeout_s: int) -> ToolResult:
        ...

    def edit_file(self, path: str, instruction: str, *, timeout_s: int) -> ToolResult:
        ...

    def find_files(self, pattern: str, *, path: str = ".", timeout_s: int) -> ToolResult:
        ...

    def grep(self, pattern: str, *, path: str = ".", file_glob: str | None = None, timeout_s: int) -> ToolResult:
        ...

    def apply_patch(self, unified_diff: str, *, timeout_s: int) -> ToolResult:
        ...


class DockerSandboxExecutor:
    def __init__(self, *, container_name: str, workspace_dir: Path, workdir: str = "/app") -> None:
        self.container_name = container_name
        self.workdir = workdir
        self.guard = PathGuard(workspace_dir)

    def shell(self, command: str, *, timeout_s: int) -> ToolResult:
        return _run_tool(
            ["docker", "exec", "-w", self.workdir, self.container_name, "bash", "-lc", command],
            timeout_s=timeout_s,
        )

    def list_dir(self, path: str, *, timeout_s: int) -> ToolResult:
        guarded = self._guard(path)
        return _run_tool(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "find \"$1\" -maxdepth 1 -mindepth 1 -printf '%f\\n' | sort",
                "cdc-list-dir",
                guarded,
            ],
            timeout_s=timeout_s,
        )

    def read_file(
        self,
        path: str,
        *,
        timeout_s: int,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        guarded = self._guard(path)
        if start_line is not None or end_line is not None:
            start = str(start_line or 1)
            end = str(end_line or 2147483647)
            return _run_tool(
                [
                    "docker",
                    "exec",
                    "-w",
                    self.workdir,
                    self.container_name,
                    "bash",
                    "-lc",
                    "total=$(wc -l < \"$1\" | tr -d ' '); "
                    "start=$2; end=$3; "
                    "[ \"$start\" -lt 1 ] && start=1; "
                    "[ \"$end\" -gt \"$total\" ] && end=$total; "
                    "printf 'File: %s\\nLines: %s-%s of %s\\n' \"$1\" \"$start\" \"$end\" \"$total\"; "
                    "sed -n \"${start},${end}p\" \"$1\" | nl -ba -v \"$start\" -w 1 -s ': '",
                    "cdc-read-file-range",
                    guarded,
                    start,
                    end,
                ],
                timeout_s=timeout_s,
            )
        return _run_tool(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "total=$(wc -l < \"$1\" | tr -d ' '); head_n=80; tail_n=80; "
                "printf 'File: %s\\n' \"$1\"; "
                "if [ \"$total\" -le $((head_n + tail_n)) ]; then "
                "printf 'Lines: 1-%s of %s\\n' \"$total\" \"$total\"; "
                "nl -ba -w 1 -s ': ' \"$1\"; "
                "else "
                "printf 'Lines: 1-%s of %s\\n' \"$head_n\" \"$total\"; "
                "sed -n \"1,${head_n}p\" \"$1\" | nl -ba -w 1 -s ': '; "
                "omitted=$((total - head_n - tail_n)); "
                "printf '[cdc:output_truncated] ... omitted %s lines ...\\n' \"$omitted\"; "
                "tail_start=$((total - tail_n + 1)); "
                "printf 'Lines: %s-%s of %s\\n' \"$tail_start\" \"$total\" \"$total\"; "
                "tail -n \"$tail_n\" \"$1\" | nl -ba -v \"$tail_start\" -w 1 -s ': '; "
                "next_end=$((head_n + 100)); [ \"$next_end\" -gt \"$total\" ] && next_end=$total; "
                "printf 'Use read_file(path=%s, start_line=%s, end_line=%s) to inspect the next chunk.\\n' \"$1\" $((head_n + 1)) \"$next_end\"; "
                "fi",
                "cdc-read-file",
                guarded,
            ],
            timeout_s=timeout_s,
        )

    def write_file(self, path: str, content: str, *, timeout_s: int) -> ToolResult:
        guarded = self._guard(path)
        return _run_tool(
            [
                "docker",
                "exec",
                "-i",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "mkdir -p -- \"$(dirname -- \"$1\")\" && cat > \"$1\"",
                "cdc-write-file",
                guarded,
            ],
            timeout_s=timeout_s,
            stdin=content,
        )

    def edit_file(self, path: str, instruction: str, *, timeout_s: int) -> ToolResult:
        del path, timeout_s
        return ToolResult(
            exit_code=2,
            stderr=f"edit_file instruction tools are not implemented; use write_file with full content: {instruction[:200]}",
        )

    def find_files(self, pattern: str, *, path: str = ".", timeout_s: int) -> ToolResult:
        guarded = self._guard(path)
        return _run_tool(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "find \"$1\" -type f -name \"$2\" -not -path '*/.*' | sort | sed -n '1,200p'",
                "cdc-find-files",
                guarded,
                pattern,
            ],
            timeout_s=timeout_s,
        )

    def grep(self, pattern: str, *, path: str = ".", file_glob: str | None = None, timeout_s: int) -> ToolResult:
        guarded = self._guard(path)
        include = file_glob or "*"
        return _run_tool(
            [
                "docker",
                "exec",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "grep -RIn --exclude-dir=.git --include \"$3\" -E -- \"$2\" \"$1\" | sed -n '1,200p'",
                "cdc-grep",
                guarded,
                pattern,
                include,
            ],
            timeout_s=timeout_s,
        )

    def apply_patch(self, unified_diff: str, *, timeout_s: int) -> ToolResult:
        self._guard_patch(unified_diff)
        strip = "1" if re.search(r"(?m)^(---|\+\+\+) [ab]/", unified_diff) else "0"
        return _run_tool(
            [
                "docker",
                "exec",
                "-i",
                "-w",
                self.workdir,
                self.container_name,
                "bash",
                "-lc",
                "patch --batch --forward -p\"$1\"",
                "cdc-apply-patch",
                strip,
            ],
            timeout_s=timeout_s,
            stdin=unified_diff,
        )

    def _guard(self, path: str) -> str:
        resolved = self.guard.resolve(path)
        rel = resolved.relative_to(self.guard.root).as_posix()
        return "." if rel == "." else rel

    def _guard_patch(self, unified_diff: str) -> None:
        for raw in _patch_paths(unified_diff):
            self._guard(raw)


def _patch_paths(unified_diff: str) -> set[str]:
    paths: set[str] = set()
    for line in unified_diff.splitlines():
        if line.startswith(("--- ", "+++ ")):
            path = line[4:].split("\t", 1)[0].strip()
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                path = path[2:]
            paths.add(path)
        elif line.startswith("diff --git "):
            parts = line.split()
            for path in parts[2:4]:
                if path.startswith(("a/", "b/")):
                    path = path[2:]
                paths.add(path)
    if not paths:
        raise ValueError("unified diff does not contain patch paths")
    return paths


def _run_tool(command: list[str], *, timeout_s: int, stdin: str | None = None) -> ToolResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(command, input=stdin, text=True, capture_output=True, timeout=timeout_s)
        return ToolResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\ncommand timed out after {timeout_s} seconds\n",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
