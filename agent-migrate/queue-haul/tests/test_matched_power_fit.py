import pytest

from matched_power_fit import SPECS, _validation, fit_spec


def test_repeat_gate_accepts_replicated_campaign_power():
    groups = {load: [{"power": power - 1}, {"power": power + 1}]
              for load, power in ((.2, 150), (.5, 200), (.8, 250))}
    result = _validation(groups)
    assert result["gate_passed"]
    assert result["repeat_holdout_rmse_w"] == pytest.approx(2)


def test_frozen_bootstrap_is_deduplicated_without_losing_draws():
    result = fit_spec(SPECS[0], bootstrap_samples=20)
    assert sum(result["bootstrap_curve_counts"]) == 20
    assert len(result["bootstrap_curve_counts"]) == len(
        result["phase_power"]["measured_power_bootstrap"])
