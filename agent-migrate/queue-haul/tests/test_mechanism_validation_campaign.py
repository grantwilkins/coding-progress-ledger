"""
Claim: mechanism Gantts are matched current-stack trials and represent median
request-boundary wait, rather than hand-picked extremes.

Plausible wrong implementations:
- Give replay and KV different context, bandwidth, activity, or sessions.
- Plot the fastest successful trace instead of the median repetition.
"""

import json

from mechanism_validation_campaign import SCHEDULE, make_plan, representatives


def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "queue-haul-migration-manifest-v2",
        "workload": "coding",
        "sessions": [{
            "id": "s0", "job_class": "coding", "rank": 0,
            "state_code": "C0", "turns": [{
                "time_s": 0, "input_tokens": 4096, "append_tokens": 32,
                "output_tokens": 1, "reset": False,
            }],
        }],
    }))
    return path


def test_plan_is_five_matched_active_request_pairs(tmp_path):
    plan = make_plan(manifest(tmp_path), session_id="s0")
    migrations = [row for row in plan["scenarios"]
                  if row["kind"] == "migration"]
    assert len(migrations) == 10
    assert {(row["repeat"], row["method"]) for row in migrations} == {
        (repeat, method) for repeat in range(5)
        for method in ("kv_transfer", "replay")
    }
    assert {(
        row["context_size"], row["bandwidth_mbps"], row["concurrency"],
        row["serving_concurrency"], row["activity"],
        tuple((turn["at_s"], turn["append_tokens"])
              for turn in row["request_schedule"]),
        tuple((session["session_id"], session["initial_tokens"])
              for session in row["sessions"]),
    ) for row in migrations} == {
        (28_000, 10_000, 1, 1, "one_turn",
         tuple((turn["at_s"], turn["append_tokens"]) for turn in SCHEDULE),
         (("s0", 28_000),))
    }


def test_representatives_choose_median_wait():
    rows = [
        {"method": method, "scenario_id": f"{method}-{index}",
         "request_wait_s": str(wait), "success": "True"}
        for method in ("kv_transfer", "replay")
        for index, wait in enumerate((9, 1, 5, 3, 7))
    ]
    assert representatives(rows, 5) == {
        "kv_transfer": "kv_transfer-2", "replay": "replay-2",
    }
