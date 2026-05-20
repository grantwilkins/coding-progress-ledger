from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadConfig:
    source: str = "generated"
    seed: int = 7
    jobs: int = 10_000
    classes: int = 48
    profile: str = "agentic_retained_sessions"

    def problem_kwargs(self) -> dict[str, int | str]:
        return {
            "workload_source": self.source,
            "workload_seed": self.seed,
            "workload_jobs": self.jobs,
            "workload_classes": self.classes,
            "workload_profile": self.profile,
        }

    def output_dir(self, root: Path) -> Path:
        base = root / "outputs" / "sweep"
        return base if self.source == "fixed" else base / self.label

    @property
    def label(self) -> str:
        return f"{self.source}_seed{self.seed}_sessions{self.jobs}_classes{self.classes}"


def parse_workload_config(description: str) -> WorkloadConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workload-source", choices=("fixed", "generated"), default="generated")
    parser.add_argument("--workload-seed", type=int, default=7)
    parser.add_argument("--workload-jobs", type=int, default=10_000)
    parser.add_argument("--workload-classes", type=int, default=48)
    parser.add_argument("--workload-profile", default="agentic_retained_sessions")
    args = parser.parse_args()
    return WorkloadConfig(
        args.workload_source,
        args.workload_seed,
        args.workload_jobs,
        args.workload_classes,
        args.workload_profile,
    )
