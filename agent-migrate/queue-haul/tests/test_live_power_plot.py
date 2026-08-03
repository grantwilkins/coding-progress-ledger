"""
Claim:
Power samples are averaged independently in fixed 5 s bins.

Plausible wrong implementations:
- Use a rolling average instead of fixed bins.
- Put a boundary sample in the preceding bin.
- Sum samples or mis-scale milliseconds as seconds.
- Highlight source shutdown instead of migration completion.
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


def test_power_uses_fixed_five_second_mean_bins():
    rows = [
        {"timestamp": f"2026/08/03 00:00:{second}", "power.draw [W]": watts}
        for second, watts in (("00.000", "100"), ("04.999", "200"),
                              ("05.000", "400"), ("09.999", "600"))
    ]

    times, watts = driver.bin_power(rows, driver.parse_wall(rows[0]["timestamp"]))

    assert times == pytest.approx([2.5, 7.5])
    assert watts == pytest.approx([150, 500])


def test_migration_highlight_ends_at_completion():
    marker = {"migration_start": 10, "migration_complete": 20,
              "source_stopped": 30}

    assert driver.migration_window(marker) == (10, 20)
