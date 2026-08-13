import pytest

from pack_power_calibration import fit_curve


def test_fit_curve_is_monotone_and_anchors_far_power():
    curve, metrics = fit_curve([
        {"load": .2, "power_w": 160}, {"load": .2, "power_w": 170},
        {"load": .5, "power_w": 150}, {"load": 1, "power_w": 280},
    ], 100, 2, 300)
    assert curve[0] == [0, 100]
    assert curve[-1] == [2, 300]
    assert all(a[1] <= b[1] for a, b in zip(curve, curve[1:]))
    assert metrics["mae_w"] == pytest.approx(5.625)
