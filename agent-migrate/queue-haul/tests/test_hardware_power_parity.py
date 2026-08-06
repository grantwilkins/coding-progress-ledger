"""
Claim:
Each hardware point compares deadline-admitted and achieved source-power shed
on the same 0--100% episode scale, using x only below parity and dots otherwise.

Plausible wrong implementations:
- Use the campaign-wide 100% goal instead of the policy's admitted request.
- Normalize by migrations pooled across scenarios instead of within each episode.
- Use elapsed campaign time instead of per-scenario completion timestamps.
- Include unmeasured missing scenarios as zero-achievement observations.
- Mark equality or over-delivery as an undershoot.
"""

import csv
import json
from types import SimpleNamespace

from plot_hardware_power_parity import load_network, load_policy, outcomes


def _scenario(scenario_id="s", policy="queue_haul", admitted=4):
    sessions = [{"session_id": str(i)} for i in range(8)]
    return {
        "scenario_id": scenario_id, "policy": policy, "sessions": sessions,
        "deadline_s": 10,
        "moves": [{"session_id": str(i), "deadline_admitted": i < admitted}
                  for i in range(8)],
    }


def test_policy_uses_per_episode_admitted_and_achieved_shed(tmp_path):
    scenario = _scenario()
    (tmp_path / "plan.json").write_text(json.dumps({"scenarios": [scenario]}))
    with (tmp_path / "policy_attainment.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "scenario_id", "policy", "power_attainment_fraction"))
        writer.writeheader()
        writer.writerow({"scenario_id": "s", "policy": "queue_haul",
                         "power_attainment_fraction": .375})

    assert load_policy(tmp_path)[0] == {
        "campaign": tmp_path.name, "scenario_id": "s",
        "method": "queue_haul", "requested_percent": 50,
        "achieved_percent": 37.5, "marker": "x",
    }


def test_network_uses_completion_times_and_omits_missing_cases(tmp_path):
    complete, missing = _scenario(), _scenario("missing")
    (tmp_path / "plan.json").write_text(
        json.dumps({"scenarios": [complete, missing]}))
    attempt = tmp_path / "scenarios/s/attempt-0001"
    attempt.mkdir(parents=True)
    (attempt / "decision.json").write_text(
        json.dumps({"moves": complete["moves"]}))
    ends = [4_000_000_000] * 4 + [11_000_000_000] * 4
    (attempt / "result.json").write_text(json.dumps({
        "status": "complete", "started_ns": 0,
        "requests": [{"request": {"end_ns": end}} for end in ends],
    }))
    curve = SimpleNamespace(power=lambda load: 100 + 100 * load)

    rows = load_network(tmp_path, curve, 5)

    assert len(rows) == 1
    assert rows[0]["requested_percent"] == 50
    assert rows[0]["achieved_percent"] == 50
    assert rows[0]["marker"] == "o"


def test_outcomes_classify_the_error_sign_and_equality_boundary():
    rows = [
        {"requested_percent": 50, "achieved_percent": achieved}
        for achieved in (40, 50, 60, 50)
    ]

    assert outcomes(rows) == [
        ("Below target", 1, 25),
        ("On target", 2, 50),
        ("Above target", 1, 25),
    ]
