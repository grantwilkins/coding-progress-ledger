"""
Claim:
The canonical campaign pairs every policy on one scenario, preserves fixed-method
baselines, executes the primary contract without pacing, labels assumed destination
capacity as sensitivity, and compares both greedies in the compact scaling run.

Plausible wrong implementations:
- Resample a different session population for each policy.
- Allow replay-only or KV-only to use the other migration method.
- Present the assumed destination contract as accepted measured evidence.
- Run the LP at small scale but silently label another planner as LP at 1M.
- Hold route capacity fixed while claiming an equivalent-capacity scale result.
- Execute planner pacing while labeling the result as the eager hardware contract.
"""

import csv
import json
from dataclasses import dataclass

from canonical_simulator_campaign import POLICIES, eager, run
from migration import ORDERED_EAGER_PARALLEL_V1


@dataclass(frozen=True)
class _Move:
    rate_limit_bytes_per_s: float | None
    quiesce_s: float | None


@dataclass(frozen=True)
class _Plan:
    moves: tuple[_Move, ...]


def test_primary_execution_is_unpaced_without_mutating_the_plan():
    planned = _Plan((_Move(10, 20),))

    executed = eager(planned)

    assert executed.moves == (_Move(None, None),)
    assert planned.moves == (_Move(10, 20),)


def test_campaign_pairs_policies_and_keeps_scale_claim_explicit(tmp_path):
    policy_rows, scale_rows = run(
        out=tmp_path, main_sessions=6, scale_sessions=(6, 12), seed=7,
        target_fractions=(.25, .5),
    )

    assert {row["policy"] for row in policy_rows} == set(POLICIES)
    assert len(policy_rows) == 2 * len(POLICIES)
    assert all(len({
        row["scenario_id"] for row in policy_rows
        if row["target_fraction"] == target
    }) == 1 for target in (.25, .5))
    assert len({row["scenario_id"] for row in policy_rows}) == 2
    assert len({row["requested_source_drop_w"] for row in policy_rows}) == 2
    assert all(row["kv_moves"] == 0 for row in policy_rows
               if row["policy"] == "replay_only")
    assert all(row["replay_moves"] == 0 for row in policy_rows
               if row["policy"] == "kv_only")
    assert all(row["deadline_met"] for row in policy_rows
               if row["policy"] == "queue_haul")
    assert all(row["evidence_status"] == "sensitivity" for row in policy_rows)
    assert [row["sessions"] for row in scale_rows] == [6, 6, 12, 12]
    assert {row["planner"] for row in scale_rows} == {
        "greedy", "greedy_lagrangian",
    }
    assert {row["topology"] for row in scale_rows} == {"pooled_destination"}
    assert all(row["deadline_met"] for row in scale_rows)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["execution_contract"] == ORDERED_EAGER_PARALLEL_V1
    assert metadata["destination_contract"].endswith("assumed sensitivity")
    assert metadata["scale_policies"] == ["greedy", "greedy_lagrangian"]
    assert metadata["scale_topology"] == "equivalent pooled destination"
    assert metadata["scale_target_fraction"] == .1
    assert metadata["scale_route_contract"] == "10 Gbps per 10K sessions"
    assert metadata["planner_expected_growth"] is False
    assert metadata["sampled_requests_enabled"] is False
    assert len(metadata["model_profile"]["sha256"]) == 64
    assert len(metadata["workload_profile"]["sha256"]) == 64
    assert list(csv.DictReader(
        (tmp_path / "representative_schedule.csv").open()
    ))
