"""
Claim:
The timeline preserves the common measured clock, pairs each continuation with
its session, distinguishes copy completion from commit, and uses source power.

Plausible wrong implementations:
- Reset each session to its own start time.
- Treat bulk-copy completion as the route-switch commit.
- Pair continuations by row order instead of session ID.
- Plot destination rather than source GPU power.
- Accept a scenario that did not run four concurrent KV writes.
"""

import csv
import json

import pytest

from plot_testbed_kv_timeline import extract, write


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path):
    root = tmp_path / "run"
    run = root / "scenarios" / "measured"
    run.mkdir(parents=True)
    (run / "scenario.json").write_text(json.dumps({
        "method": "kv_transfer", "concurrency": 4,
    }))
    migrations = []
    continuations = []
    for order in range(4):
        session = f"s{order}"
        start = 1_000_000_000 + order * 100_000_000
        migrations.append({
            "scenario_id": "measured", "session_id": session,
            "order": order, "method": "kv_transfer", "concurrency": 4,
            "bandwidth_mbps": 1000, "activity": "one_turn", "success": "True",
            "initial_start_ns": start, "initial_end_ns": start + 2_000_000_000,
            "pause_start_ns": start + 2_000_000_000,
            "catch_up_start_ns": start + 2_500_000_000,
            "catch_up_end_ns": start + 3_500_000_000,
            "switch_start_ns": start + 3_500_000_000,
            "switch_end_ns": start + 4_000_000_000,
        })
        continuations.insert(0, {
            "session_id": session, "start_ns": start + 4_100_000_000,
            "first_byte_ns": start + 4_300_000_000,
            "end_ns": start + 4_500_000_000,
        })
    _csv(root / "migrations.csv", migrations)
    (run / "result.json").write_text(json.dumps({"continuations": continuations}))
    _csv(run / "power.csv", [
        {"monotonic_ns": 1_000_000_000, "gpu": gpu, "power_w": power,
         "valid": 1}
        for gpu, power in ((1, 80), (0, 300))
    ])
    return root


def test_extract_preserves_measured_event_semantics(tmp_path):
    timeline, power = extract(fixture(tmp_path), "measured")
    assert timeline[0]["bulk_start_s"] == 0
    assert timeline[1]["bulk_start_s"] == pytest.approx(.1)
    assert timeline[0]["bulk_finish_s"] == 2
    assert timeline[0]["bulk_s"] == 2
    assert timeline[0]["commit_s"] == 4
    assert timeline[0]["route_switch_s"] == pytest.approx(.5)
    assert timeline[0]["first_token_s"] == pytest.approx(4.3)
    assert timeline[0]["commit_to_first_token_s"] == pytest.approx(.3)
    assert timeline[0]["continuation_ttft_s"] == pytest.approx(.2)
    assert timeline[3]["first_token_s"] == pytest.approx(4.6)
    assert power == [{
        "scenario_id": "measured", "time_s": 0, "source_power_w": 300,
        "source_gpu": 0, "evidence_status": "measured",
        "provenance": str(tmp_path / "run/scenarios/measured/power.csv"),
    }]


def test_write_rejects_wrong_concurrency_and_emits_plot(tmp_path):
    root = fixture(tmp_path)
    scenario = root / "scenarios/measured/scenario.json"
    scenario.write_text(json.dumps({"method": "kv_transfer", "concurrency": 2}))
    with pytest.raises(ValueError, match="concurrency 4"):
        extract(root, "measured")
    scenario.write_text(json.dumps({"method": "kv_transfer", "concurrency": 4}))
    out = tmp_path / "out"
    write(root, "measured", out)
    assert (out / "kv_write_concurrency_4_timeline.csv").stat().st_size
    assert (out / "kv_write_concurrency_4_timeline.png").stat().st_size
    assert (out / "kv_write_concurrency_4_timeline.pdf").stat().st_size
