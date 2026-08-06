"""
Claim:
Power samples are averaged independently in fixed 1 s bins.

Plausible wrong implementations:
- Use a rolling average instead of fixed bins.
- Put a boundary sample in the preceding bin.
- Sum samples or mis-scale milliseconds as seconds.
- Include samples before the requested display window.
- Include samples at or beyond the display cutoff.
- Highlight source shutdown instead of migration completion.
- End switching before destination steady state.
- Start shutdown before destination steady state.
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


def test_power_uses_fixed_one_second_mean_bins():
    rows = [
        {"timestamp": f"2026/08/03 00:00:{second}", "power.draw [W]": watts}
        for second, watts in (("00.000", "100"), ("00.999", "200"),
                              ("01.000", "400"), ("01.999", "600"))
    ]

    times, watts = driver.bin_power(rows, driver.parse_wall(rows[0]["timestamp"]))

    assert times == pytest.approx([.5, 1.5])
    assert watts == pytest.approx([150, 500])
    assert driver.bin_power(rows, driver.parse_wall(rows[0]["timestamp"]), 1) \
        == ([.5], [500])
    assert driver.bin_power(rows, driver.parse_wall(rows[0]["timestamp"]), 0, 1) \
        == ([.5], [150])


def test_migration_highlight_ends_at_completion():
    marker = {"migration_start": 10, "migration_complete": 20,
              "destination_steady": 30, "source_stopped": 40}

    assert driver.migration_window(marker) == (10, 20)
    assert driver.switch_window(marker) == (20, 30)
    assert driver.shutdown_window(marker) == (30, 40)
