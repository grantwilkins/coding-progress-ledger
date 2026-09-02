import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

import model_hardware_drain_campaign as campaign


def test_a100_command_runs_all_three_arms_end_to_end(monkeypatch, tmp_path):
    models = iter(sorted(campaign.MODELS["A100"]))
    monkeypatch.setattr(campaign, "_profile", lambda path, hardware: (
        SimpleNamespace(model=next(models)), path))
    monkeypatch.setattr(campaign, "_snapshot", lambda *_args: None)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(campaign, "reduce",
                        lambda roots, out, expected: {
                            "roots": roots, "out": out, "expected": expected})

    result = campaign.run(
        "A100", [tmp_path / f"p{i}.json" for i in range(3)],
        tmp_path / "cluster.json", tmp_path / "calibration.json",
        tmp_path / "manifest.json", tmp_path / "run", tmp_path / "key")

    assert len(calls) == 21
    assert all(call[1]["check"] and call[1]["env"]["QH_RUNTIME"] == "native"
               for call in calls)
    assert all("drain" in call[0] for call in calls[:3])
    assert all("--stack-block" in call[0] for call in calls[3:18])
    assert [call[1]["env"]["QH_MODEL_PROFILE"] for call in calls[3:6]] \
        != [call[1]["env"]["QH_MODEL_PROFILE"] for call in calls[6:9]]
    assert all("reduce" in call[0] for call in calls[18:])
    assert result["roots"] == [(tmp_path / "run").resolve()]
    assert result["expected"] == {
        (model, "A100") for model in campaign.MODELS["A100"]}


def test_h100_is_a_separate_complete_command(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "_profile", lambda path, hardware: (
        SimpleNamespace(model="openai/gpt-oss-20b"), path))
    monkeypatch.setattr(campaign, "_snapshot", lambda *_args: None)
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(campaign, "reduce", lambda *_args: {})

    campaign.run("H100", [tmp_path / "profile.json"],
                 *(tmp_path / name for name in
                   ("cluster", "calibration", "manifest", "run", "key")))

    assert len(calls) == 7
    assert sum("--stack-block" in call for call in calls) == 5


def test_profile_requires_the_adjacent_passing_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    path = tmp_path / "profile.json"
    path.write_text("{}")
    profile = SimpleNamespace(
        model="openai/gpt-oss-20b", hardware="H100", precision="BF16",
        tensor_parallel=1, kv_geometry=object())
    monkeypatch.setattr(campaign.ModelProfile, "load", lambda _path: profile)
    gate = {"schema": "queue-haul-model-architecture-gate-v1",
            "model": profile.model, "hardware": "H100",
            "passed": False, "launch": {"passed": True},
            "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".gate.json").write_text(json.dumps(gate))

    with pytest.raises(ValueError, match="gated BF16"):
        campaign._profile(path, "H100")
    gate["passed"] = True
    path.with_suffix(".gate.json").write_text(json.dumps(gate))
    assert campaign._profile(path, "H100") == (profile, path.relative_to(tmp_path))


def test_reduce_writes_arm_table_and_both_canonical_figures(monkeypatch, tmp_path):
    arm = tmp_path / "run/arms/gpt"
    arm.mkdir(parents=True)
    profile = campaign.ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
    (arm / "profile.json").write_bytes(profile.read_bytes())
    monkeypatch.setattr(campaign, "_gated_profile",
                        lambda path, _hardware: campaign.ModelProfile.load(path))
    plan = {
        "design": "drain", "model_profile": {"path": str(profile),
            "sha256": hashlib.sha256(profile.read_bytes()).hexdigest()},
        "manifest": {"sha256": "m"},
        "cluster": {"destinations": [
            {"id": "east", "region": "eastus2"},
            {"id": "germany", "region": "germanywestcentral"}]},
        "scenarios": [{"condition_index": index % 10, "repeat": index // 10,
                       "sessions": [{"initial_tokens": 100}]}
                      for index in range(50)],
    }
    plan_path = arm / "plan.json"
    plan_path.write_text(json.dumps(plan))
    (arm / "run_metadata.json").write_text(json.dumps({
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "runtime_environment": {"QH_RUNTIME": "native",
                                "QH_LMCACHE_MODE": "mp"}}))
    (arm / "summary.json").write_text(json.dumps({
        "expected": 50, "completed": 49, "failed": 1, "missing": 0,
        "valid": False}))
    rows = [{
        "status": "failed" if index == 0 else "complete",
        "attempt": "1", "excluded_attempts": "0",
        "modeled_power_attainment_s": str(10 + index / 10),
        "time_to_target_s": "" if index == 0 else str(5 + index / 10),
        "target_met": str(index > 0),
        "modeled_power_deadline_met": str(index > 0), "east_replay": "1",
        "east_kv_transfer": "2", "germany_replay": "2",
        "germany_kv_transfer": "3",
    } for index in range(50)]
    with (arm / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = campaign.reduce(
        [tmp_path / "run"], tmp_path / "out",
        {("openai/gpt-oss-20b", "A100")})

    assert summary["openai/gpt-oss-20b / A100"] == {
        "episodes": 50, "completed_episodes": 49, "failed_episodes": 1,
        "action_mix_episodes": 50,
        "retried_episodes": 0, "excluded_attempts": 0,
        "drain_deadline_attainment": .98,
        "modeled_power_deadline_attainment": .98}
    assert len(list(csv.DictReader(
        (tmp_path / "out/drain_episodes.csv").open()))) == 50
    for name in ("drain_attainment_ecdf", "drain_action_mix"):
        assert (tmp_path / "out" / f"{name}.png").is_file()
        assert (tmp_path / "out" / f"{name}.pdf").is_file()
    with pytest.raises(ValueError, match="incomplete arm set"):
        campaign.reduce([tmp_path / "run"], tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
