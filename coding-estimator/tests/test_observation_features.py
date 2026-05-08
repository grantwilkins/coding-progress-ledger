"""
Claim:
- observation features consume only observation events at or before the
  checkpoint step.
- verifier terminal events do not leak backward into earlier prefixes.
"""

from __future__ import annotations

import json
from pathlib import Path

from types import SimpleNamespace

from coding_estimator.checkpoints.features import observation
from coding_estimator.ingest.run_record import RunRecord


def _run(tmp_path: Path) -> RunRecord:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "ledger.jsonl").write_text("")
    rows = [
        {
            "schema_version": "0.1.0",
            "run_id": "demo",
            "step": 1,
            "observed_ts": "2026-05-05T00:00:01Z",
            "source_artifact": "transcript.jsonl",
            "event_type": "solution_oracle_read",
            "payload": {},
        },
        {
            "schema_version": "0.1.0",
            "run_id": "demo",
            "step": 2,
            "observed_ts": "2026-05-05T00:00:02Z",
            "source_artifact": "transcript.jsonl",
            "event_type": "validation_attempt",
            "payload": {"after_solution_oracle_read": True},
        },
        {
            "schema_version": "0.1.0",
            "run_id": "demo",
            "step": 3,
            "observed_ts": "2026-05-05T00:00:03Z",
            "source_artifact": "transcript.jsonl",
            "event_type": "agent_claims_done",
            "payload": {},
        },
        {
            "schema_version": "0.1.0",
            "run_id": "demo",
            "step": 4,
            "observed_ts": "2026-05-05T00:00:04Z",
            "source_artifact": "run_manifest.json",
            "event_type": "verifier_disagreement",
            "payload": {},
        },
    ]
    (run_dir / "observation_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return RunRecord(
        run_id="demo",
        source="tb_live_v2",
        ledger_path=run_dir / "ledger.jsonl",
        events=(SimpleNamespace(step=0), SimpleNamespace(step=3)),  # type: ignore[arg-type]
        has_real_wallclock=True,
        start_wall_time=None,
        end_wall_time=None,
        task_id="demo_task",
        task_family="demo",
        arm="A",
        difficulty="easy",
        agent_scaffold=None,
        model_name=None,
        raw_metadata={},
    )


def test_observation_features_are_prefix_safe(tmp_path: Path) -> None:
    run = _run(tmp_path)
    step2 = observation.compute(2, run)
    step3 = observation.compute(3, run)

    assert step2["obs_solution_oracle_read_so_far"] is True
    assert step2["obs_num_validation_attempts_so_far"] == 1
    assert step2["obs_agent_claims_done_so_far"] is False
    assert step2["obs_verifier_disagreement_proxy_so_far"] is False

    assert step3["obs_agent_claims_done_so_far"] is True
    assert step3["obs_done_without_validation_so_far"] is False
    # verifier_disagreement synthetic terminal event sits at step 4, so
    # it is not visible at checkpoint 3.
    assert step3["obs_verifier_disagreement_proxy_so_far"] is False
