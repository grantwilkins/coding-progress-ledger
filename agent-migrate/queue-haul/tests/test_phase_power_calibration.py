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


def test_run_plan_resumes_completed_cells(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "queue-haul-phase-power-plan-v1", "cells": [
        {"mixture": "prefill", "target_service_load": .1, "repeat": 0},
        {"mixture": "decode", "target_service_load": .2, "repeat": 0}]}))
    profile = tmp_path / "profile.json"; profile.write_text("{}")
    out = tmp_path / "run"; out.mkdir()
    fields = ["mixture", "repeat", "target_service_load", "f_tps", "g_tps",
              "power_mean_w"]
    with (out / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow(dict(zip(fields, ("prefill", 0, .1, 1, 0, 100))))
    monkeypatch.setattr(calibration.ModelProfile, "load", lambda _: type(
        "P", (), {"case": lambda self: type("C", (), {"F": 1, "G": 1})()})())
    monkeypatch.setattr(calibration.power_rate_sweep, "validate_gpu", lambda *args: None)
    monkeypatch.setattr(calibration, "run_cell", lambda host, port, root, cell, F, G: {
        "mixture": cell["mixture"], "repeat": cell["repeat"],
        "target_service_load": cell["target_service_load"], "f_tps": 0,
        "g_tps": 1, "power_mean_w": 110})
    result = calibration.run_plan(plan, profile, out, resume=True)
    assert [row["mixture"] for row in result] == ["prefill", "decode"]


def test_fit_accepts_compact_identifiable_group_holdouts():
    compact = []
    for mixture, ratio in (("prefill75", .25), ("mixed", 1), ("decode", 4)):
        for load in (.2, .5, .8):
            f, g = 1000 * load, 1000 * load * ratio
            z = .001 * f + .002 * g
            compact.append({"mixture": mixture, "repeat": 0, "f_tps": f,
                            "g_tps": g, "power_mean_w": 100 + 200 * z / (1 + z)})
    assert calibration.fit(compact, 100, bootstrap_samples=2)["gate_passed"]
