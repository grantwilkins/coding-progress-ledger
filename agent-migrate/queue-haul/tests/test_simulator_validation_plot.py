"""
Claim:
The validation plot compares simulator events with independent exact values for
shared-link completion, commit time, and source power.

Plausible wrong implementations:
- Serialize equal transfers instead of sharing the link.
- Let aggregate transfer rate exceed link capacity.
- Lower source power when transfer finishes instead of at commit.
- Plot simulator output without checking independent expected values.
"""

import csv

import pytest

from plot_simulator_validation import _rows, validation_result, write


def test_validation_case_matches_hand_calculation():
    rows = {row["quantity"]: row for row in _rows(validation_result())}
    assert all(row["simulated"] == row["expected"] for row in rows.values())
    assert rows["session_0_transfer_start_s"]["simulated"] == 0
    assert rows["session_1_transfer_start_s"]["simulated"] == 0
    assert rows["session_0_transfer_complete_s"]["simulated"] == 2
    assert rows["session_1_transfer_complete_s"]["simulated"] == 2
    assert rows["session_0_commit_s"]["simulated"] == 3
    assert rows["session_1_commit_s"]["simulated"] == 3
    assert rows["source_power_drop_s"]["simulated"] == 3


def test_validation_plot_writes_checked_evidence(tmp_path):
    write(tmp_path)
    rows = list(csv.DictReader((tmp_path / "simulator_validation.csv").open()))
    assert all(
        float(row["simulated"]) == pytest.approx(float(row["expected"])) for row in rows
    )
    assert (tmp_path / "simulator_validation.png").stat().st_size > 0
    assert (tmp_path / "simulator_validation.pdf").stat().st_size > 0
