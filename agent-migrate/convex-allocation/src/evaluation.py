from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadConfig:
    source: str = "fixed"
    seed: int = 7
    jobs: int = 1000
    classes: int = 12
    profile: str = "shed_event_long_context"

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
        return f"{self.source}_seed{self.seed}_jobs{self.jobs}_classes{self.classes}"


def parse_workload_config(description: str) -> WorkloadConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workload-source", choices=("fixed", "generated"), default="fixed")
    parser.add_argument("--workload-seed", type=int, default=7)
    parser.add_argument("--workload-jobs", type=int, default=1000)
    parser.add_argument("--workload-classes", type=int, default=12)
    parser.add_argument("--workload-profile", default="shed_event_long_context")
    args = parser.parse_args()
    return WorkloadConfig(
        args.workload_source,
        args.workload_seed,
        args.workload_jobs,
        args.workload_classes,
        args.workload_profile,
    )
