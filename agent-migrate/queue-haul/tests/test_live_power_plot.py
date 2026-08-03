"""
Claim:
Power samples are averaged independently in fixed 500 ms bins.

Plausible wrong implementations:
- Use a rolling average instead of fixed bins.
- Put a boundary sample in the preceding bin.
- Sum samples or mis-scale milliseconds as seconds.
"""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "live_power_driver",
    Path(__file__).parents[1] / "outputs/live-power-oneoff/driver.py",
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_power_uses_fixed_half_second_mean_bins():
    rows = [
        {"timestamp": f"2026/08/03 00:00:0{second}", "power.draw [W]": watts}
        for second, watts in (("0.000", "100"), ("0.499", "200"),
                              ("0.500", "400"), ("0.999", "600"))
    ]

    times, watts = driver.bin_power(rows, driver.parse_wall(rows[0]["timestamp"]))

    assert times == pytest.approx([.25, .75])
    assert watts == pytest.approx([150, 500])
