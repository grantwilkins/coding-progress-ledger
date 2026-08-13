import pytest

from pack_power_calibration import baseline_gate, bootstrap_curves, fit_curve, grouped_repeat_cv


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
