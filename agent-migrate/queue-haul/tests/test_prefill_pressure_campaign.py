import csv
import json
from collections import Counter

import pytest

import constrained_state_campaign as campaign
import prefill_pressure_campaign as pressure


@pytest.fixture
def plan(monkeypatch, tmp_path):
    source = {"frozen": campaign.freeze_inputs(campaign.default_inputs(), 7)}
    monkeypatch.setattr(pressure.contention, "calibrate", lambda *args: {
        "replay_speedup": 1, "kv_tail_s": 1})
    monkeypatch.setattr(pressure, "background_response", lambda *args: {
        "baseline_work_per_rps": [.36, .012], "replay_duration_per_rho": 1,
        "kv_tail_s_per_rho": 1})
    monkeypatch.setattr(pressure.contention, "simulate", lambda *args: {
        "relief_w": 0, "target_time_s": None})
    return pressure.prepare(tmp_path, source, source, [], source, [])


def test_fixed_pressure_brackets_survive_predicted_failures(plan, tmp_path):
    original = campaign.default_inputs()
    inputs = plan["frozen"]["inputs"]
    pack = next(row for row in inputs["packs"] if row["pack_id"] == plan["design"]["selected_pack"])
    template = next(row for row in original["packs"] if row["pack_id"] == "large-r0")
    assert len(pack["sessions"]) == 8
    assert {row["initial_tokens"] for row in pack["sessions"]} == {27360}
    for before, after in zip(template["sessions"], pack["sessions"]):
        assert after == {**before, "initial_tokens": 27360, "log_bytes": 54720}
    assert pack["power_gains"] == template["power_gains"]
    for key in ("profile", "background_manifest", "target"):
        assert inputs[key] == original[key]
    assert plan["frozen"]["constants"]["power_window_s"] == 5
    assert len(plan["schedule"]) == 60
    assert Counter((row["n_prefill"], row["policy"]) for row in plan["schedule"]) == {
        (level, policy): 3 for level in (0, .4, .5, .6) for policy in campaign.POLICIES}
    assert {row["repeat"] for row in plan["schedule"]} == {0, 1, 2}
    assert {(row["wan_mbps"], row["deadline_s"], row["n_hbm"], row["n_serving"])
            for row in plan["states"]} == {(4000, 30, 0, 0)}
    with (tmp_path / "simulation.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    assert {float(row["n_prefill"]) for row in predictions} == {0, .4, .5, .6}
    assert all(float(row["relief_w"]) == 0 and row["target_time_s"] == ""
               for row in predictions)
    assert campaign._verify_plan(plan) == plan["sha256"]


@pytest.mark.parametrize("control_fails", (False, True))
def test_pressure_claim_requires_full_control_and_windowed_loaded_attainment(plan, tmp_path, control_fails):
    pack = next(row for row in plan["frozen"]["inputs"]["packs"]
                if row["pack_id"] == plan["design"]["selected_pack"])
    raw = []
    for job in plan["schedule"]:
        late = job["policy"] == "per_session_greedy" and (job["n_prefill"] or control_fails)
        raw.append({"episode_id": job["episode_id"], "capacity_inputs": {},
                    "decisions": [{"session_id": session["session_id"],
                                   "action": "replay" if job["policy"] == "replay_only" else "kv_transfer",
                                   "completion_s": 28 if late else 20}
                                  for session in pack["sessions"]]})
    (tmp_path / "raw_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw) + "\n")
    if control_fails:
        with pytest.raises(ValueError, match="no prefill-induced transition"):
            pressure.summarize(plan, tmp_path)
    else:
        assert pressure.summarize(plan, tmp_path)["robust"]
    validation = json.loads((tmp_path / "prefill_validation.json").read_text())
    assert validation["control_full"] is not control_fails
    assert validation["genuine_prefill_transition"] is not control_fails
    assert validation["robust"] is not control_fails
    assert all(row["attained"]["per_session_greedy"] == 0
               for row in validation["cases"] if row["n_prefill"])
    with (tmp_path / "episodes.csv").open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 60
    assert all(float(row["achieved_relief_w"]) == pytest.approx(pack["power_gains"][-1])
               for row in episodes)
