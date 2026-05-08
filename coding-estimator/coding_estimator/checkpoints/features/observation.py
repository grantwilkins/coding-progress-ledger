"""Observation-channel features derived from transcript/verifier events."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from coding_estimator.ingest.run_record import RunRecord
from coding_estimator.runner.observation_events import (
    build_observation_events,
    read_observation_events,
)

GROUP = "observation"
COLUMNS: tuple[str, ...] = (
    "obs_num_validation_attempts_so_far",
    "obs_num_validation_successes_so_far",
    "obs_num_validation_failures_so_far",
    "obs_num_error_observations_so_far",
    "obs_num_repeated_errors_so_far",
    "obs_num_environment_blocks_so_far",
    "obs_solution_oracle_read_so_far",
    "obs_agent_claims_done_so_far",
    "obs_done_without_validation_so_far",
    "obs_verifier_disagreement_proxy_so_far",
    "obs_validation_after_oracle_read_so_far",
)


@lru_cache(maxsize=512)
def _load_events(run_dir_text: str, task_id: str | None) -> tuple[dict[str, Any], ...]:
    run_dir = Path(run_dir_text)
    path = run_dir / "observation_events.jsonl"
    if path.is_file():
        rows = read_observation_events(path)
    else:
        rows = build_observation_events(run_dir=run_dir, run_id=run_dir.name, task_id=task_id)
    return tuple(rows)


def compute(t_step: int, run: RunRecord) -> dict[str, Any]:
    rows = _load_events(str(run.ledger_path.parent), run.task_id)
    visible = [row for row in rows if int(row.get("step", 0)) <= t_step]

    validation_attempts = 0
    validation_successes = 0
    validation_failures = 0
    error_count = 0
    repeated_errors = 0
    environment_blocks = 0
    oracle_read = False
    agent_done = False
    validation_after_oracle = False

    for row in visible:
        event_type = str(row.get("event_type"))
        payload = row.get("payload") or {}
        if event_type == "validation_attempt":
            validation_attempts += 1
            if bool(payload.get("after_solution_oracle_read")):
                validation_after_oracle = True
        elif event_type == "validation_pass_observed":
            validation_successes += 1
        elif event_type == "validation_fail_observed":
            validation_failures += 1
        elif event_type == "error_observed":
            error_count += 1
        elif event_type == "error_repeated":
            repeated_errors += 1
        elif event_type == "environment_blocked":
            environment_blocks += 1
        elif event_type == "solution_oracle_read":
            oracle_read = True
        elif event_type == "agent_claims_done":
            agent_done = True

    done_without_validation = agent_done and validation_attempts == 0
    verifier_disagreement_proxy = agent_done and done_without_validation

    return {
        "obs_num_validation_attempts_so_far": validation_attempts,
        "obs_num_validation_successes_so_far": validation_successes,
        "obs_num_validation_failures_so_far": validation_failures,
        "obs_num_error_observations_so_far": error_count,
        "obs_num_repeated_errors_so_far": repeated_errors,
        "obs_num_environment_blocks_so_far": environment_blocks,
        "obs_solution_oracle_read_so_far": oracle_read,
        "obs_agent_claims_done_so_far": agent_done,
        "obs_done_without_validation_so_far": done_without_validation,
        "obs_verifier_disagreement_proxy_so_far": verifier_disagreement_proxy,
        "obs_validation_after_oracle_read_so_far": validation_after_oracle,
    }
