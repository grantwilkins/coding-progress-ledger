import csv
import json
from types import SimpleNamespace

import pytest

import constrained_state_campaign as campaign
import contention_campaign as contention
from profiles import ModelProfile


def test_calibration_uses_successful_fixed_policy_timings_and_rejects_failures():
    inputs = campaign.default_inputs()
    pack, case = inputs["packs"][0], ModelProfile.load(campaign.DEFAULT_PROFILE).case()
    state = campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0]
    schedule = campaign.execution_schedule([state], [pack], 7, repeats=1)
    plan = {"frozen": campaign.freeze_inputs(inputs, 7), "schedule": schedule}
    raw = []
    for job in schedule:
        if job["policy"] not in ("kv_only", "replay_only"):
            continue
        replay = job["policy"] == "replay_only"
        elapsed = sum((row["initial_tokens"] / case.replay.rate(row["initial_tokens"], 1)
                       + case.replay_completion_s) / 2 if replay else
                      case.kv_transfer.sealed_bytes(row["initial_tokens"]) / 1_250_000_000
                      for row in pack["sessions"]) + (0 if replay else 3)
        raw.append({"episode_id": job["episode_id"],
                    "decisions": [{"session_id": row["session_id"],
                                   "completion_s": elapsed}
                                  for row in pack["sessions"]]})
    result = contention.calibrate(plan, raw)
    assert result["replay_speedup"] == pytest.approx(2)
    assert result["kv_tail_s"] == pytest.approx(3)
    assert len(result["evidence"]) == 2
    raw[0]["decisions"][0]["completion_s"] = None
    with pytest.raises(ValueError, match="calibration migration failed"):
        contention.calibrate(plan, raw)


@pytest.mark.parametrize("action, completion", (("kv_transfer", 3), ("replay", 2.4)))
def test_simulation_shares_capacity_and_credits_only_completed_power(action, completion):
    pack = {"sessions": [{"session_id": str(i), "initial_tokens": 10} for i in range(2)],
            "power_gains": [0, 1, 1, 2]}
    case = SimpleNamespace(replay=SimpleNamespace(rate=lambda tokens, count: 10),
                           replay_completion_s=0,
                           kv_transfer=SimpleNamespace(sealed_bytes=lambda tokens: 100))
    profile = SimpleNamespace(case=lambda: case)
    moves = [{"session_id": row["session_id"], "action": action} for row in pack["sessions"]]
    calibration = {"replay_speedup": 1, "kv_tail_s": 1}
    result = contention.simulate(pack, moves, .0008, 6, profile, calibration)
    assert list(result["completion_times"].values()) == pytest.approx([completion] * 2)
    assert result["relief_w"] == pytest.approx(2 * (6 - completion) / 5)
    assert result["target_time_s"] is None
    assert contention.simulate(pack, [], .0008, 6, profile, calibration)["relief_w"] == 0


def test_summary_retains_baseline_wins_ties_and_failed_actions(tmp_path):
    inputs = campaign.default_inputs()
    pack = inputs["packs"][0]
    state = {**campaign.state_grid(dict.fromkeys(campaign.AXES, 0), [10000])[0],
             "deadline_s": 14}
    schedule = campaign.execution_schedule([state], [pack], 7, repeats=3)
    plan = {"frozen": campaign.freeze_inputs(inputs, 7), "states": [state],
            "schedule": schedule}
    raw = []
    for job in schedule:
        failed = job["policy"] in ("replay_only", "per_session_greedy")
        raw.append({"episode_id": job["episode_id"], "capacity_inputs": {},
                    "decisions": [{"session_id": row["session_id"],
                                   "action": "replay" if job["policy"] == "replay_only" else "kv_transfer",
                                   "completion_s": None if failed else
                                   1 if job["policy"] == "kv_only" else 12,
                                   "error": "request failed" if failed else None}
                                  for row in pack["sessions"]]})
    (tmp_path / "raw_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(ValueError, match="all outcomes retained"):
        contention.summarize(plan, tmp_path)
    with (tmp_path / "comparisons.csv").open() as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 18
    assert all(row["result"] == ("loss" if row["baseline"] == "kv_only" else "win")
               for row in comparisons)
    assert not json.loads((tmp_path / "validation.json").read_text())["all_QH_no_losses"]
    for episode in raw:
        for move in episode["decisions"]:
            move.update(completion_s=None, error="request failed")
    (tmp_path / "raw_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(ValueError, match="all outcomes retained"):
        contention.summarize(plan, tmp_path)
    with (tmp_path / "comparisons.csv").open() as handle:
        assert {row["result"] for row in csv.DictReader(handle)} == {"tie"}
    for episode, job in zip(raw, schedule):
        for move in episode["decisions"]:
            move.update(completion_s=1 if job["policy"] in ("queue_haul", "greedy") else 12,
                        error=None)
    (tmp_path / "raw_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw) + "\n")
    assert {row["result"] for row in contention.summarize(plan, tmp_path)} == {"win"}
    validation = json.loads((tmp_path / "validation.json").read_text())
    assert all(validation[key] for key in ("complete", "three_repeats_per_case_policy",
                                          "all_QH_no_losses", "strict_win_vs_per_session"))
