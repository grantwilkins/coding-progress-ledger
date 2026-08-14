import csv
import json

import pytest

import evidence_catalog as evidence


def test_catalog_labels_model_credit_and_checks_hash(tmp_path):
    root = tmp_path / "hardware-gap-001"
    root.mkdir()
    result = root / "result.json"
    result.write_text("{}")
    value = evidence.catalog(tmp_path)
    entry = evidence.verify(value, tmp_path, result)
    assert entry["evidence_kind"] == "model_credited"
    result.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="checksum changed"):
        evidence.verify(value, tmp_path, result)


def test_trailing_power_uses_exact_windows(tmp_path):
    run = tmp_path / "hardware-gap-001"
    stack = run / "stacks" / "s"
    attempt = run / "scenarios" / "id" / "attempt-0001"
    stack.mkdir(parents=True); attempt.mkdir(parents=True)
    with (stack / "power.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("monotonic_ns", "wall_ns", "gpu", "power_w",
                         "utilization_pct", "memory_mib", "valid"))
        for tenth in range(0, 111):
            seconds = tenth / 10
            watts = 200 if seconds < 5 else 150
            writer.writerow((int(seconds * 1e9), 0, 0, watts, 0, 0, 1))
    (attempt / "scenario.json").write_text(json.dumps({
        "scenario_id": "id", "condition_id": "all-bind", "policy": "p",
        "repeat": 0, "deadline_s": 5,
    }))
    (attempt / "result.json").write_text(json.dumps({
        "started_ns": 5_000_000_000, "realized_shed_w": 40,
    }))
    (attempt / "source_load.jsonl").write_text(json.dumps({
        "start_ns": 10_000_000_000,
    }) + "\n")
    rows = evidence.trailing_power_rows(run)
    assert len(rows) == 1
    assert rows[0]["measured_trailing_shed_w"] == 50
    assert rows[0]["modeled_deadline_shed_w"] == 40
