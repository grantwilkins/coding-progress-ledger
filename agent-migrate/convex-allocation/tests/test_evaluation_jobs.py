"""
Claim:
Experiment job execution returns rows in input-grid order while allowing
independent jobs to run in a bounded process pool.

Plausible wrong implementations:
- Return rows in completion order instead of grid order.
- Ignore CONVEX_ALLOCATION_WORKERS.
- Let the default worker count exceed the intended cap.
"""

from __future__ import annotations

import time

import evaluation
from evaluation import _worker_count, run_jobs


def _delayed_identity(job):
    value, delay = job
    time.sleep(delay)
    return value


def test_run_jobs_process_pool_preserves_input_order(monkeypatch):
    monkeypatch.setenv("CONVEX_ALLOCATION_WORKERS", "2")

    rows = run_jobs("ordered", ((1, 0.05), (2, 0.0), (3, 0.0)), _delayed_identity)

    assert rows == [1, 2, 3]


def test_worker_count_respects_override_and_default_cap(monkeypatch):
    monkeypatch.setenv("CONVEX_ALLOCATION_WORKERS", "3")
    assert _worker_count(10) == 3
    assert _worker_count(2) == 2

    monkeypatch.delenv("CONVEX_ALLOCATION_WORKERS")
    monkeypatch.setattr(evaluation.os, "cpu_count", lambda: 64)
    assert _worker_count(20) == 8
