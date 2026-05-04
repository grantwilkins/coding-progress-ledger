"""P_step produces a strictly-increasing, contiguous step list and is
the only policy implemented in v0.

Claim:
    checkpoint_steps(run, P_STEP) returns range(s_min, s_max+1) where
    s_min and s_max are the smallest and largest event steps in the run.
    Every other policy raises NotImplementedError.

Plausible wrong implementations:
    - return only the distinct event steps (skipping gaps) -> not contiguous
    - skip the init step (start at s_min+1) -> off-by-one feature window
    - off-by-one on the terminal step (s_max-1 or s_max+1)
    - silently fall back to P_step for an unimplemented policy
"""

from __future__ import annotations

import pytest

from coding_estimator.checkpoints.policy import (
    V0_IMPLEMENTED,
    CheckpointPolicy,
    checkpoint_steps,
    p_step_checkpoints,
)
from coding_estimator.ingest.run_record import RunRecord


def _fake_run(steps: list[int]) -> RunRecord:
    """Build a minimal RunRecord with events whose steps are `steps`.
    We don't need real LedgerEvents for the policy: it only reads
    `e.step`. Use a tiny stand-in."""
    from types import SimpleNamespace

    events = tuple(SimpleNamespace(step=s) for s in steps)
    return RunRecord(  # type: ignore[arg-type]
        run_id="r0",
        source="tb_live",
        ledger_path=None,  # type: ignore[arg-type]
        events=events,  # type: ignore[arg-type]
        has_real_wallclock=False,
        start_wall_time=None,
        end_wall_time=None,
        task_id=None,
        task_family=None,
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


def test_p_step_strictly_increasing_and_contiguous() -> None:
    run = _fake_run([0, 1, 1, 2, 4, 4, 7])  # gaps at 3, 5, 6
    out = p_step_checkpoints(run)
    assert out == [0, 1, 2, 3, 4, 5, 6, 7]
    assert all(out[i + 1] - out[i] == 1 for i in range(len(out) - 1))


def test_p_step_includes_first_and_last_step() -> None:
    run = _fake_run([2, 5, 7])
    out = p_step_checkpoints(run)
    assert out[0] == 2 and out[-1] == 7


def test_p_step_empty_run_raises() -> None:
    run = _fake_run([])
    with pytest.raises(ValueError, match="no events"):
        p_step_checkpoints(run)


def test_unimplemented_policy_raises_not_implemented() -> None:
    run = _fake_run([0, 1])
    for policy in CheckpointPolicy:
        if policy in V0_IMPLEMENTED:
            continue
        with pytest.raises(NotImplementedError):
            checkpoint_steps(run, policy)


def test_default_policy_is_p_step() -> None:
    run = _fake_run([0, 1, 2])
    assert checkpoint_steps(run) == p_step_checkpoints(run)


def test_p_step_on_golden_fixture_covers_every_step() -> None:
    """The golden fixture has events from step 0 to step 13. P_step must
    return exactly [0,1,...,13]; an off-by-one would skip step 0 (init)
    or fail to include step 13 (terminal)."""
    from pathlib import Path

    from ledger_progress.serialization import load_events_jsonl

    fixture = Path(__file__).parent / "fixtures" / "golden_run" / "ledger.jsonl"
    events = load_events_jsonl(str(fixture))
    run = _fake_run([e.step for e in events])
    assert p_step_checkpoints(run) == list(range(0, 14))
