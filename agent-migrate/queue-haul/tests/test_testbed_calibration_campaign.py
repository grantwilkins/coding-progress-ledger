import csv

import testbed_calibration_campaign as campaign


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def test_reducer_enforces_timing_and_operational_gates(tmp_path):
    timing = [{"predicted_s": 10, "observed_s": 10.5,
               "correctness_failures": 0} for _ in range(9)]
    service, migration, operational = (tmp_path / name for name in (
        "service.csv", "migration.csv", "operational.csv"))
    write(service, timing); write(migration, timing)
    write(operational, [{"state": state, "repeat": repeat,
                         "predicted_feasible": True, "observed_feasible": True,
                         "correctness_failures": 0}
                        for state in campaign.STATES for repeat in range(3)])
    result = campaign.reduce(service, migration, operational)
    assert result["service_gate_passed"]
    assert result["migration_gate_passed"]
    assert result["operational_gate_passed"]
