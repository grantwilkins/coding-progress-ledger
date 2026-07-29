"""
Claim:
The ECDF uses migration start to first content token, preserves every measured
migration, separates physical cells, and classifies foreground overlap.

Plausible wrong implementations:
- Start timing at the destination request and omit KV prefetch.
- Use the first SSE chunk instead of the first content token.
- Pool different context/bandwidth cells.
- Miss a request active when migration starts.
- Aggregate or weight repeated observations before constructing the ECDF.
"""

import json

import pytest

from plot_migration_ttft_cdf import ecdf, extract, write


def _case(root, method, start, request_start, first, foreground):
    path = root / "rho0.8-t16384-b10000-r0" / method
    path.mkdir(parents=True)
    (path / "scenario.json").write_text(json.dumps({"concurrency": 1}))
    (path / "result.json").write_text(json.dumps({
        "migrations": [{
            "move": {"method": method},
            "initial_start_ns": start,
            "switch_end_ns": start + 9_000_000_000,
            "initial": {
                "start_ns": request_start,
                "first_byte_ns": first,
                "end_ns": first + 1_000_000_000,
                "stream_chunks": [{"monotonic_ns": request_start + 1}],
            },
            "error": None,
        }],
    }))
    (path / "foreground").mkdir()
    (path / "foreground/requests.json").write_text(json.dumps(foreground))


def test_extract_uses_migration_start_content_token_and_overlap(tmp_path):
    _case(tmp_path, "kv_transfer", 1_000_000_000, 3_000_000_000,
          6_000_000_000, [{"start_ns": 0, "end_ns": 2_000_000_000}])
    _case(tmp_path, "replay", 10_000_000_000, 10_100_000_000,
          12_000_000_000,
          [{"start_ns": 11_000_000_000, "end_ns": 13_000_000_000}])

    rows = {row["method"]: row for row in extract(tmp_path)}

    assert rows["kv_transfer"]["migration_ttft_s"] == 5
    assert rows["kv_transfer"]["transfer_or_replay_s"] == 2
    assert rows["kv_transfer"]["destination_request_ttft_s"] == 3
    assert rows["kv_transfer"]["foreground_overlap"] == "active_at_start"
    assert rows["replay"]["migration_ttft_s"] == 2
    assert rows["replay"]["foreground_overlap"] == "arrived_during"


def test_ecdf_preserves_each_observation_and_write_emits_figure(tmp_path):
    x, y = ecdf([4, 1, 2])
    assert x.tolist() == [1, 2, 4]
    assert y.tolist() == pytest.approx([1 / 3, 2 / 3, 1])

    _case(tmp_path, "kv_transfer", 1, 2, 4, [])
    _case(tmp_path, "replay", 1, 2, 5, [])
    out = tmp_path / "out"
    write(tmp_path, out)
    for name in (
        "migration_ttft_cdf.csv",
        "migration_ttft_cdf.png",
        "migration_ttft_cdf.pdf",
    ):
        assert (out / name).stat().st_size
