"""
Claim:
The scaling plot pairs identical experiments and reports simulated deadline
power, selected moves, and completed moves with the correct denominators.

Plausible wrong implementations:
- Plot planned power instead of simulated deadline-window power.
- Divide completed moves by all sessions instead of selected moves.
- Compare solvers at different session counts or power targets.
- Label a new experiment with the previous experiment's deadline.
"""

import csv

import pytest

from plot_scaling_results import plot_title, ratios, read_rows


def test_scaling_ratios_use_simulated_power_and_matching_denominators():
    row = {
        "sessions": 10,
        "planned_moves": 8,
        "moves_completed_by_deadline": 6,
        "requested_source_drop_w": 50,
        "planned_source_drop_w": 45,
        "modeled_source_drop_at_deadline_w": 40,
    }
    assert ratios(row) == pytest.approx((80, 75, 80))


def test_scaling_title_uses_the_recorded_deadline():
    assert plot_title({
        "bandwidth_gbps_per_node": 1,
        "deadline_s": 900,
        "target_fraction_of_removable_power": 0.5,
    }) == "Coding, 1 Gbps/node, 15 min deadline, 50% awake-state power reduction"


def test_scaling_rows_require_paired_solver_settings(tmp_path):
    fields = ("solver", "sessions", *(
        "source_instances", "source_nodes", "bandwidth_gbps_per_node", "deadline_s",
        "end_s", "target_fraction_of_removable_power", "requested_source_drop_w",
    ))
    rows = [
        ("node_aware", 10, 1, 1, 1, 120, 180, 0.5, 50),
        ("lp", 10, 1, 1, 1, 120, 180, 0.5, 51),
    ]
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(rows)
    with pytest.raises(ValueError, match="settings differ"):
        read_rows(path)
