"""
Claim:
The service-surface reducer computes per-level throughput and power from bundle
timestamps, not from whole-run aggregates.

Plausible wrong implementations:
- Assign every request in the bundle to every level.
- Divide token counts by total run duration instead of the level window.
- Average power outside the manifest level window.
- Silently accept malformed bundles.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import service_profile_reduce as r


def write_bundle(root: Path, probe_type: str = "prefill_staircase") -> Path:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    manifest = {
        "tp": 1,
        "gpus_per_node": 1,
        "probe": {
            "type": probe_type,
            "levels": [
                {
                    "level": 0,
                    "label": "a",
                    "concurrency": 1,
                    "t_start_epoch": 10.0,
                    "t_end_epoch": 20.0,
                    "params": {"input_len": 100, "output_len": 10, "prefix_len": 0},
                },
                {
                    "level": 1,
                    "label": "b",
                    "concurrency": 2,
                    "t_start_epoch": 20.0,
                    "t_end_epoch": 30.0,
                    "params": {"input_len": 200, "output_len": 20, "prefix_len": 0},
                },
            ],
        },
    }
    requests = {
        "input_lens": [100, 200],
        "output_lens": [10, 20],
        "ttfts": [0.1, 0.2],
        "itls": [[0.01, 0.02], [0.03, 0.04]],
        "request_timestamps": [15.0, 25.0],
    }
    power = "timestamp,power.draw [W]\n15,50\n25,70\n"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "requests.json").write_text(json.dumps(requests))
    (bundle / "power.csv").write_text(power)
    return bundle


def test_read_power_parses_nvidia_smi_local_wall_clock(tmp_path: Path):
    epoch = 1783368731.25
    stamp = datetime.fromtimestamp(epoch).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    path = tmp_path / "power.csv"
    path.write_text("timestamp, power.draw [W]" + chr(10) + f"{stamp}, 50 W" + chr(10))

    ts, watts = r.read_power(path)

    assert ts[0] == pytest.approx(epoch, abs=0.001)
    assert watts[0] == pytest.approx(50.0)


def test_level_row_uses_level_window_for_requests_power_and_rates(tmp_path: Path):
    bundle = write_bundle(tmp_path)
    rows = r.read_rows(bundle)

    assert [x["n_requests"] for x in rows] == [1, 1]
    assert rows[0]["input_tps"] == pytest.approx(10.0)
    assert rows[1]["input_tps"] == pytest.approx(20.0)
    assert rows[0]["output_tps"] == pytest.approx(1.0)
    assert rows[1]["output_tps"] == pytest.approx(2.0)
    assert rows[0]["power_mean_w"] == pytest.approx(50.0)
    assert rows[1]["power_mean_w"] == pytest.approx(70.0)
    assert rows[0]["ttft_p50_ms"] == pytest.approx(100.0)
    assert rows[1]["tpot_p95_ms"] == pytest.approx(39.5)


def test_discover_bundles_accepts_parent_directory(tmp_path: Path):
    bundle = write_bundle(tmp_path / "run")

    assert r.discover_bundles(tmp_path / "run") == [bundle]


def test_malformed_bundle_hard_fails(tmp_path: Path):
    bundle = tmp_path / "bad"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    (bundle / "requests.json").write_text("{}")
    (bundle / "power.csv").write_text("timestamp,power.draw [W]\n")

    with pytest.raises(ValueError):
        r.read_rows(bundle)


def test_service_scale_rows_normalize_rho_and_best_decode_G():
    rows = [
        {"probe_type": "prefill_staircase", "input_len": 100, "concurrency": 1, "input_tps": 50.0, "output_tps": 0.0, "power_mean_w": 10.0},
        {"probe_type": "prefill_staircase", "input_len": 200, "concurrency": 1, "input_tps": 25.0, "output_tps": 0.0, "power_mean_w": 12.0},
        {"probe_type": "decode_staircase", "input_len": 100, "concurrency": 1, "input_tps": 0.0, "output_tps": 100.0, "power_mean_w": 20.0},
        {"probe_type": "decode_staircase", "input_len": 100, "concurrency": 2, "input_tps": 0.0, "output_tps": 150.0, "power_mean_w": 22.0},
        {"probe_type": "decode_staircase", "input_len": 200, "concurrency": 1, "input_tps": 0.0, "output_tps": 75.0, "power_mean_w": 21.0},
    ]

    out = r.service_scale_rows(rows)

    assert [(x["metric"], x["input_len"]) for x in out] == [("rho", 100), ("rho", 200), ("G", 100), ("G", 200)]
    assert [x["throughput_tps"] for x in out] == pytest.approx([50.0, 25.0, 150.0, 75.0])
    assert [x["scale_vs_short"] for x in out] == pytest.approx([1.0, 0.5, 1.0, 0.5])
    assert out[2]["concurrency"] == 2
