"""Checkpoint policy: where in each run a checkpoint row exists.

v0 ships only `P_step` (one checkpoint per ledger step) for parity with
upstream `build_estimator_checkpoints.py`. The enum lists alternatives
documented in CHECKPOINT_POLICY.md but not implemented; reintroduce
when there is a concrete reason to compare fidelity.
"""

from __future__ import annotations

from enum import Enum

from coding_estimator.ingest.run_record import RunRecord


class CheckpointPolicy(Enum):
    P_STEP = "P_step"
    # Documented but not implemented in v0:
    P_EVENT = "P_event"
    P_KSTEP = "P_kstep"
    P_WALLCLOCK_GRID = "P_wallclock_grid"
    P_TERMINAL_ONLY = "P_terminal_only"


V0_DEFAULT = CheckpointPolicy.P_STEP
V0_IMPLEMENTED: frozenset[CheckpointPolicy] = frozenset({CheckpointPolicy.P_STEP})


def p_step_checkpoints(run: RunRecord) -> list[int]:
    """Return the sequence of checkpoint steps for `P_step` on this run.

    One checkpoint per distinct ledger step from the smallest step in the
    run (typically 0, the init event) through the largest. Strictly
    increasing; never empty for a valid run.
    """
    if not run.events:
        raise ValueError(f"run {run.source}/{run.run_id} has no events")
    steps = sorted({e.step for e in run.events})
    return list(range(steps[0], steps[-1] + 1))


def checkpoint_steps(run: RunRecord, policy: CheckpointPolicy = V0_DEFAULT) -> list[int]:
    if policy not in V0_IMPLEMENTED:
        raise NotImplementedError(
            f"policy {policy.value!r} is documented in CHECKPOINT_POLICY.md "
            "but not implemented in v0; only P_step ships."
        )
    return p_step_checkpoints(run)
