"""
Claim:
The timeline preserves one measured clock, keeps inference active through the
pause, and distinguishes copy completion, commit, and post-switch inference.

Plausible wrong implementations:
- Reset inference and migration events to different time origins.
- Stop source inference when the pause begins instead of draining active work.
- Treat bulk-copy completion as the route-switch commit.
- Pair a continuation with the wrong session.
- Accept a scenario whose KV-write concurrency is not one.
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
        "method": "kv_transfer", "concurrency": 1,
    }))
    _csv(root / "migrations.csv", [{
        "scenario_id": "measured", "session_id": "s0", "order": 0,
        "method": "kv_transfer", "concurrency": 1,
        "bandwidth_mbps": 1000, "activity": "staged", "success": "True",
        "initial_start_ns": 1_000_000_000,
        "initial_end_ns": 3_000_000_000,
        "pause_start_ns": 3_000_000_000,
        "catch_up_start_ns": 4_000_000_000,
        "catch_up_end_ns": 5_000_000_000,
        "switch_start_ns": 5_000_000_000,
        "switch_end_ns": 5_100_000_000,
    }])
    (run / "result.json").write_text(json.dumps({
        "activities": [
            {"session_id": "s0", "stage_index": 0,
             "start_ns": 1_100_000_000, "first_byte_ns": 1_600_000_000,
             "end_ns": 1_800_000_000},
            {"session_id": "s0", "stage_index": 1,
             "start_ns": 2_100_000_000, "first_byte_ns": 3_200_000_000,
             "end_ns": 3_500_000_000},
        ],
        "continuations": [{
            "session_id": "s0", "start_ns": 5_200_000_000,
            "first_byte_ns": 5_400_000_000,
            "end_ns": 5_600_000_000,
        }],
    }))
    return root


def test_extract_preserves_pause_and_inference_semantics(tmp_path):
    timeline, segments = extract(fixture(tmp_path), "measured")
    row = timeline[0]
    assert row["bulk_start_s"] == 0
    assert row["bulk_finish_s"] == 2
    assert row["quiesce_s"] == 2
    assert row["commit_s"] == pytest.approx(4.1)
    assert row["first_token_s"] == pytest.approx(4.4)
    assert [(segment["phase"], segment["start_s"], segment["finish_s"])
            for segment in segments] == [
        ("<P>", .1, .6), ("<D>", .6, .8), ("<Tool>", .8, 1.1),
        ("<P>", 1.1, 2.2), ("<D>", 2.2, 2.5),
        ("<P>", 4.2, 4.4), ("<D>", 4.4, 4.6),
    ]
    assert max(
        segment["finish_s"] for segment in segments
        if segment["location"] == "source"
    ) > row["quiesce_s"]


def test_write_rejects_wrong_concurrency_and_emits_plot(tmp_path):
    root = fixture(tmp_path)
    scenario = root / "scenarios/measured/scenario.json"
    scenario.write_text(json.dumps({"method": "kv_transfer", "concurrency": 2}))
    with pytest.raises(ValueError, match="concurrency 1"):
        extract(root, "measured")
    scenario.write_text(json.dumps({"method": "kv_transfer", "concurrency": 1}))
    out = tmp_path / "out"
    write(root, "measured", out)
    assert (out / "kv_write_concurrency_1_timeline.csv").stat().st_size
    assert (out / "kv_write_concurrency_1_inference.csv").stat().st_size
    assert (out / "kv_write_concurrency_1_timeline.png").stat().st_size
    assert (out / "kv_write_concurrency_1_timeline.pdf").stat().st_size
