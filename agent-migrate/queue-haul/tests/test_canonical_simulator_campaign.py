"""
Claim:
The canonical campaign pairs every policy on one scenario, preserves fixed-method
baselines, labels assumed destination capacity as sensitivity, and changes only
session count in the compact scaling run.

Plausible wrong implementations:
- Resample a different session population for each policy.
- Allow replay-only or KV-only to use the other migration method.
- Present the assumed destination contract as accepted measured evidence.
- Run the LP at small scale but silently label another planner as LP at 1M.
- Hold route capacity fixed while claiming an equivalent-capacity scale result.
"""

import csv
import json

from canonical_simulator_campaign import POLICIES, run


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
    assert [row["sessions"] for row in scale_rows] == [6, 12]
    assert {row["planner"] for row in scale_rows} == {"queue_haul_greedy"}
    assert {row["topology"] for row in scale_rows} == {"pooled_destination"}
    assert all(row["deadline_met"] for row in scale_rows)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["destination_contract"].endswith("assumed sensitivity")
    assert metadata["scale_policy"] == "queue_haul_greedy"
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
