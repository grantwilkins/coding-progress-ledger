from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def run_jobs(label, jobs, fn):
    jobs = tuple(jobs)
    if not jobs:
        return []
    workers = _worker_count(len(jobs))
    if workers == 1:
        return [_progress(label, i + 1, len(jobs), fn(job)) for i, job in enumerate(jobs)]
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            _log_progress(label, done, len(jobs))
    return results


def _worker_count(job_count):
    requested = int(os.environ.get("CONVEX_ALLOCATION_WORKERS", "0") or 0)
    if requested <= 0:
        requested = min(8, os.cpu_count() or 1)
    return max(1, min(requested, job_count))


def _progress(label, done, total, result):
    _log_progress(label, done, total)
    return result


def _log_progress(label, done, total):
    print(f"{label}: {done}/{total}", file=sys.stderr, flush=True)
