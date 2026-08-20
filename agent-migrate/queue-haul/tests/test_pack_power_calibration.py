"""
Claim:
The fixed-pack fit is monotone, resamples whole repeats, rejects unstable
baselines, and records validation quality for the measured curve it installs.

Plausible wrong implementations:
- Resample individual windows and leak repeats across validation folds.
- Inherit quality scores from the phase model replaced by the measured curve.
- Report in-sample error as the installed model's uncertainty.
"""

import json

import pytest

import pack_power_calibration as calibration
from pack_power_calibration import baseline_gate, bootstrap_curves, fit_curve, grouped_repeat_cv
from test_profiles import profile, write


def test_fit_curve_is_monotone_and_anchors_far_power():
    curve, metrics = fit_curve([
        {"load": .2, "power_w": 160}, {"load": .2, "power_w": 170},
        {"load": .5, "power_w": 150}, {"load": 1, "power_w": 280},
    ], 100, 2, 300)
    assert curve[0] == [0, 100]
    assert curve[-1] == [2, 300]
    assert all(a[1] <= b[1] for a, b in zip(curve, curve[1:]))
    assert metrics["mae_w"] == pytest.approx(5.625)


def test_bootstrap_curves_resample_whole_repeats():
    points = [{"repeat": repeat, "load": load, "power_w": power + repeat}
              for repeat in range(3) for load, power in ((.2, 150), (.5, 200))]
    curves = bootstrap_curves(points, 100, 1, 300, samples=5)
    assert len(curves) == 5
    assert all(curve[0] == [0, 100] and curve[-1] == [1, 300] for curve in curves)
    cv = grouped_repeat_cv(points, 100, 1, 300)
    assert cv["grouped_repeat_cv_mae_w"] == pytest.approx(1)
    assert cv["grouped_repeat_cv_within_5w_fraction"] == 1


def test_baseline_gate_rejects_only_nonsteady_full_pack_windows():
    rows = [{"scenario_id": str(i), "baseline_source_power_w": value}
            for i, value in enumerate((278, 279, 280, 281, 170))]
    rejected, summary = baseline_gate(rows)
    assert rejected == {"4"}
    assert summary["baseline_median_w"] == 279


def test_fit_records_the_installed_curve_validation(tmp_path, monkeypatch):
    raw = profile()
    raw["schema"] = "queue-haul-model-profile-v5"
    raw["max_power_load"] = 1
    raw["cases"]["central"]["phase_power"] = {
        "p0_w": 100, "delta_w": 100,
        "a_s_per_prefill_token": .001, "b_s_per_decode_token": .01,
        "valid_hull": [[0, 0], [1000, 0], [0, 100]],
        "grouped_cv_rmse_w": 99, "within_5w_fraction": 0,
        "bootstrap": [[100, 100, .001, .01]],
        "provenance_sha256": "0" * 64,
    }
    source = write(tmp_path, raw, "phase.json")
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "trailing_power.csv").write_text("power\n")
    points = [
        {"repeat": repeat, "load": load, "power_w": watts + repeat}
        for repeat in range(3) for load, watts in ((.2, 150), (.5, 200))
    ]
    monkeypatch.setattr(calibration, "collect", lambda *_: (points, {}))
    out, summary_path, points_path = (
        tmp_path / "pack.json", tmp_path / "summary.json", tmp_path / "points.csv")

    summary = calibration.fit(source, root, out, summary_path, points_path, 300)
    fitted = json.loads(out.read_text())
    phase = fitted["cases"]["central"]["phase_power"]

    assert phase["grouped_cv_rmse_w"] == pytest.approx(
        summary["grouped_repeat_cv_rmse_w"])
    assert phase["within_5w_fraction"] == pytest.approx(
        summary["grouped_repeat_cv_within_5w_fraction"])
    assert fitted["sources"]["power"]["relative_error"] == pytest.approx(
        summary["grouped_repeat_cv_mae_w"] / 200)
