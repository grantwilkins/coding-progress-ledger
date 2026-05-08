from __future__ import annotations

from pathlib import Path

from .base import AgentBudget, AgentResult
from coding_data_collection.recording.transcript_recorder import RunRecorder
from coding_data_collection.sandbox.docker_executor import SandboxExecutor


class ShellCommandBackend:
    name = "shell_command"
    collection_kind = "substrate_smoke"

    def __init__(self, command: str) -> None:
        self.command = command

    def run(
        self,
        *,
        run_id: str,
        task_md: str,
        workspace_dir: Path,
        sandbox: SandboxExecutor,
        budget: AgentBudget,
        recorder: RunRecorder,
    ) -> AgentResult:
        del run_id, task_md, workspace_dir
        result = sandbox.shell(self.command, timeout_s=budget.max_tool_time_s)
        recorder.record(
            "shell",
            summary="shell_command_backend",
            command=self.command,
            exit_code=result.exit_code,
            stdout_snippet=result.stdout[-2000:],
            stderr_snippet=result.stderr[-2000:],
            duration_ms=result.duration_ms,
            visible_to_agent=True,
        )
        recorder.record("done", summary="shell command backend complete", visible_to_agent=True)
        return AgentResult(
            completed=result.exit_code == 0,
            termination_reason="agent_done" if result.exit_code == 0 else "agent_command_failed",
            steps_used=recorder.step,
        )
