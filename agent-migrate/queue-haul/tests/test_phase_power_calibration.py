import pytest
import csv
import json

import phase_power_calibration as calibration


def rows():
    result = []
    p0, delta, a, b = 98, 202, .001, .01
    for mixture, fraction in zip(calibration.MIXTURES, (1, .75, .5, .25, 0)):
        for repeat in range(3):
            for load in (.2, .5, 1):
                f, g = 1000 * load * fraction, 100 * load * (1 - fraction)
                z = a * f + b * g
                result.append({"mixture": mixture, "repeat": repeat,
                               "f_tps": f, "g_tps": g,
                               "power_mean_w": p0 + delta * z / (1 + z)})
    return result


def test_fit_recovers_phase_coefficients_and_grouped_gate():
    fitted = calibration.fit(rows(), 98, bootstrap_samples=5)
    assert fitted["gate_passed"]
    assert fitted["delta_w"] == pytest.approx(202, rel=1e-4)
    assert fitted["a_s_per_prefill_token"] == pytest.approx(.001, rel=1e-4)
    assert fitted["b_s_per_decode_token"] == pytest.approx(.01, rel=1e-4)
    assert fitted["grouped_cv_rmse_w"] < 1e-4


def test_plan_is_balanced_minimum():
    plan = calibration.campaign_plan()
    assert len(plan["cells"]) == 5 * 6 * 3
    assert {row["measurement_s"] for row in plan["cells"]} == {30}
    assert plan["idle_measurement_s"] == 30


def test_decode_load_is_paced_instead_of_batched():
    _prompt, output_tokens, rate, batch = calibration._shape(0, .5, 1000, 400)
    assert (output_tokens, rate, batch) == (512, 400 * .5 / 512, 0)


def test_run_plan_resumes_completed_cells(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "queue-haul-phase-power-plan-v1", "cells": [
        {"mixture": "prefill", "target_service_load": .1, "repeat": 0},
        {"mixture": "decode", "target_service_load": .2, "repeat": 0}]}))
    out = tmp_path / "run"; out.mkdir()
    (out / "metadata.json").write_text(json.dumps({
        "schema": "queue-haul-phase-power-run-v1", "model": "model",
        "hardware": "h100", "F_prefill_tps": 1, "G_decode_tps": 1,
        "plan_sha256": calibration.hashlib.sha256(plan.read_bytes()).hexdigest(),
    }, sort_keys=True))
    fields = ["mixture", "repeat", "target_service_load", "f_tps", "g_tps",
              "power_mean_w"]
    with (out / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow(dict(zip(fields, ("prefill", 0, .1, 1, 0, 100))))
    monkeypatch.setattr(calibration.power_rate_sweep, "validate_gpu", lambda *args: None)
    monkeypatch.setattr(calibration, "run_cell", lambda host, port, root, cell, F, G, model: {
        "mixture": cell["mixture"], "repeat": cell["repeat"],
        "target_service_load": cell["target_service_load"], "f_tps": 0,
        "g_tps": 1, "power_mean_w": 110})
    result = calibration.run_plan(plan, None, out, resume=True, model="model",
                                  hardware="h100", F=1, G=1)
    assert [row["mixture"] for row in result] == ["prefill", "decode"]


def test_explicit_h100_target_is_frozen_in_metadata(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "queue-haul-phase-power-plan-v1",
                                "cells": []}))
    seen = []
    monkeypatch.setattr(calibration.power_rate_sweep, "validate_gpu",
                        lambda *args: seen.append(args))

    calibration.run_plan(plan, None, tmp_path / "run", model="model",
                         hardware="h100", F=2, G=3)

    assert seen == [("NVIDIA H100 NVL", 400)]
    metadata = json.loads((tmp_path / "run/metadata.json").read_text())
    assert (metadata["model"], metadata["F_prefill_tps"],
            metadata["G_decode_tps"]) == ("model", 2, 3)


def test_idle_anchor_rejects_work_and_persists_power(tmp_path, monkeypatch):
    values = iter(((1, 2, 3), (1, 2, 3)))
    monkeypatch.setattr(calibration, "_metrics", lambda *_: next(values))
    monkeypatch.setattr(calibration.time, "sleep", lambda _: None)
    monkeypatch.setattr(calibration.power_rate_sweep, "power",
                        lambda path, _stop, _interval: path.write_text(
                            "monotonic_ns,wall_ns,power_w,utilization_pct,memory_mib\n"
                            + "".join(f"{i},0,100,0,1\n" for i in range(7))))
    ticks = iter((0, 10))
    monkeypatch.setattr(calibration.time, "monotonic_ns", lambda: next(ticks))

    row = calibration.measure_idle("host", 1, tmp_path, 0, 1)

    assert row["power_mean_w"] == 100
    assert json.loads((tmp_path / "idle.jsonl").read_text())["sequence"] == 0


def test_suite_checkpoints_and_skips_completed_target(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "queue-haul-phase-power-plan-v1",
                                "cells": [{"mixture": "mixed",
                                           "target_service_load": .5,
                                           "repeat": 0}]}))
    targets = tmp_path / "targets.json"
    target = {"name": "model", "model": "model", "hardware": "h100",
              "prefill_tps": 2, "decode_tps": 3, "vllm": "vllm"}
    targets.write_text(json.dumps([target]))
    calls = []
    def run(_plan, out, model, hardware, F, G, vllm, host, port, resume):
        calls.append(resume); out.mkdir(parents=True)
        (out / "metadata.json").write_text(json.dumps({
            "plan_sha256": calibration.hashlib.sha256(plan.read_bytes()).hexdigest(),
            "model": model, "hardware": hardware,
            "F_prefill_tps": F, "G_decode_tps": G}))
        (out / "measurements.csv").write_text(
            "mixture,target_service_load,repeat\nmixed,0.5,0\n")
        (out / "idle.jsonl").write_text("{}\n{}\n")
    monkeypatch.setattr(calibration, "run_with_server", run)

    calibration.run_suite(plan, targets, tmp_path / "out")
    calibration.run_suite(plan, targets, tmp_path / "out")

    assert calls == [False]


def test_fit_accepts_compact_identifiable_group_holdouts():
    compact = []
    for mixture, ratio in (("prefill75", .25), ("mixed", 1), ("decode", 4)):
        for load in (.2, .5, .8):
            f, g = 1000 * load, 1000 * load * ratio
            z = .001 * f + .002 * g
            compact.append({"mixture": mixture, "repeat": 0, "f_tps": f,
                            "g_tps": g, "power_mean_w": 100 + 200 * z / (1 + z)})
    assert calibration.fit(compact, 100, bootstrap_samples=2)["gate_passed"]


def test_fit_holds_out_validation_groups():
    compact = rows()
    for row in compact:
        row["validation_group"] = {"prefill": "prefill_heavy",
            "prefill75": "prefill_heavy", "decode75": "mixed"}.get(
                row["mixture"], row["mixture"])
    fitted = calibration.fit(compact, 98, bootstrap_samples=2)
    assert fitted["groups"] == ["decode", "mixed", "prefill_heavy"]
