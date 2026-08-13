import pytest

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
