from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coding_data_collection.recording.transcript_recorder import RunRecorder
from coding_data_collection.sandbox.docker_executor import SandboxExecutor


@dataclass(frozen=True)
class AgentBudget:
    max_steps: int = 80
    max_wall_time_s: int = 1800
    max_tool_time_s: int = 120
    min_steps_before_done: int = 0
    require_validation_before_done: bool = False
    allow_blocked_done: bool = True


@dataclass(frozen=True)
class AgentResult:
    completed: bool
    termination_reason: str
    steps_used: int
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float | None = None


class AgentBackend(Protocol):
    name: str
    collection_kind: str

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
        ...
