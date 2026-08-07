from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "outputs/live-power-shed/driver.py"
SPEC = importlib.util.spec_from_file_location("live_power_shed", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_power_pairing_uses_query_interval_and_drops_one_torn_tick(tmp_path):
    path = tmp_path / "power_100ms.csv"
    path.write_text(
        "timestamp,query.start,query.end,uuid,power.draw [W]\n"
        "2026/01/01 00:00:00.100,2026/01/01 00:00:00.050,2026/01/01 00:00:00.150,a,100\n"
        "2026/01/01 00:00:00.100,2026/01/01 00:00:00.050,2026/01/01 00:00:00.150,b,110\n"
        "2026/01/01 00:00:00.100,2026/01/01 00:00:00.060,2026/01/01 00:00:00.160,a,101\n"
        "2026/01/01 00:00:00.100,2026/01/01 00:00:00.060,2026/01/01 00:00:00.160,b,111\n"
        "2026/01/01 00:00:00.200,2026/01/01 00:00:00.170,2026/01/01 00:00:00.230,b,112\n"
    )

    rows, by_uuid = driver.load_power(tmp_path, ["a", "b"])

    assert len(rows) == 2
    assert [len(by_uuid[uuid]) for uuid in ("a", "b")] == [1, 1]


def test_power_pairing_rejects_multiple_incomplete_ticks(tmp_path):
    (tmp_path / "power_100ms.csv").write_text(
        "timestamp,query.start,query.end,uuid,power.draw [W]\n"
        "2026/01/01 00:00:00.100,2026/01/01 00:00:00.050,2026/01/01 00:00:00.150,a,100\n"
        "2026/01/01 00:00:00.200,2026/01/01 00:00:00.170,2026/01/01 00:00:00.230,b,110\n"
    )

    with pytest.raises(RuntimeError, match="incomplete ticks"):
        driver.load_power(tmp_path, ["a", "b"])


def test_subtick_event_uses_nearest_power_samples():
    samples = [(index / 10, float(index)) for index in range(20)]

    assert driver.phase_mean(samples, .95, .95005, event=True) == pytest.approx(9.5)
    with pytest.raises(RuntimeError, match="power phase"):
        driver.phase_mean(samples, .95, .95005)


def test_engine_queue_summary_accounts_for_depth(tmp_path):
    path = tmp_path / "engine.csv"
    path.write_text(
        "monotonic_ns,vllm:num_requests_running,vllm:num_requests_waiting\n"
        "1000000000,2,3\n2000000000,4,7\n3000000000,6,11\n"
    )

    assert driver.engine_queue(path, 1, 2) == {
        "running": {"mean": 3, "max": 4},
        "waiting": {"mean": 5, "max": 7},
    }
    with pytest.raises(RuntimeError, match="no engine samples"):
        driver.engine_queue(path, 4, 5)


def test_busy_sessions_have_unique_compute_heavy_turns():
    session = driver.destination.Session("s", 4, 2, 3, 100, 0)

    busy = driver.busy_sessions([session])[0]

    assert busy.append_tokens == driver.BUSY_APPEND_TOKENS == 2048
    assert busy.output_tokens == driver.BUSY_OUTPUT_TOKENS == 32
    assert busy.prefix_tokens == session.prefix_tokens


def test_live_scenario_has_no_reset_or_inline_verification():
    scenario = driver.live_scenario({"sessions": [{}, {}]})

    assert scenario["reset_caches"] is False
    assert scenario["verify_continuations"] is False
    assert scenario["wait_cache_idle"] is False
    assert scenario["warm_on_move"] is False
    assert scenario["prestage_all"] is True
    assert scenario["warm_concurrency"] == 8
    assert scenario["sample_power"] is False
    assert scenario["final_state"] == "awake"
    assert scenario["deadline_s"] == driver.MIGRATION_DEADLINE_S
    assert driver.MIGRATION_DEADLINE_S == 30
    assert driver.SOURCE_INFLIGHT == 64
    assert "flush_lmcache" not in PATH.read_text()


def test_traffic_switch_stops_source_and_releases_prepared_takeover():
    calls = []

    class Load:
        def __init__(self, name):
            self.name = name

        def pause(self):
            calls.append((self.name, "pause"))

        def resume(self):
            calls.append((self.name, "resume"))

    class Markers:
        def add(self, event):
            calls.append(("marker", event))

    driver.switch_traffic(Load("source"), Load("takeover"), Markers())

    assert calls == [
        ("marker", "switch_start"),
        ("takeover", "resume"),
        ("source", "pause"),
        ("marker", "traffic_switched"),
    ]
