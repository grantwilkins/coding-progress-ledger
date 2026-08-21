import pytest

from matched_power_fit import _validation


def test_repeat_gate_accepts_replicated_campaign_power():
    groups = {load: [{"power": power - 1}, {"power": power + 1}]
              for load, power in ((.2, 150), (.5, 200), (.8, 250))}
    result = _validation(groups)
    assert result["gate_passed"]
    assert result["repeat_holdout_rmse_w"] == pytest.approx(2)
