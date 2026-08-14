import json

import pytest

from frontier_model_audit import calibrate_linear_load, service_load


def row(rate, f, g, power):
    return {"rate_rps": rate, "window_s": 1, "prompt_tokens": f,
            "output_tokens": g, "power_mean_w": power}


def test_linear_load_calibration_separates_power_from_service_rates():
    prefill = [row(1, 100, 0, 20), row(2, 200, 0, 30)]
    mixed = [row(1, 100, 10, 25), row(2, 100, 20, 30)]
    result = calibrate_linear_load(prefill, mixed, 200, 10, 1)

    assert result["alpha"] == pytest.approx(.005)
    assert result["beta"] == pytest.approx(.025)
    assert result["mixed_rmse_w"] == pytest.approx(0)


def test_service_load_uses_achieved_request_starts(tmp_path):
    rows = [
        {"start_ns": 8_000_000_000, "prompt_tokens": 100, "output_tokens": 10},
        {"start_ns": 9_000_000_000, "prompt_tokens": 100, "output_tokens": 10},
        {"start_ns": 10_000_000_000, "prompt_tokens": 100, "output_tokens": 10},
    ]
    path = tmp_path / "load.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert service_load(path, 10_000_000_000, 2, 100, 10) == pytest.approx(2)
