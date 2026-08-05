import json

import plot_network_ttft_cdf as p


def scenario(root, name, attempt, status, method="kv_transfer", first=3_000_000_000):
    path = root / "scenarios" / name / attempt
    path.mkdir(parents=True)
    body = {"scenario_id": name, "status": status}
    if status == "complete":
        body |= {"started_ns": 1_000_000_000, "ended_ns": 4_000_000_000,
                 "migration_s": 3.0, "requests": [{"method": method, "request": {
                     "start_ns": 1_000_000_000, "first_byte_ns": first,
                     "end_ns": 4_000_000_000}}]}
    (path / "result.json").write_text(json.dumps(body))
    (path / "scenario.json").write_text(json.dumps({
        "bandwidth": "natural", "bandwidth_mbps": 2156.4,
        "workload": "agentic_tool_loop", "context_size": 8192}))


def test_retried_attempt_supersedes_preempted_failure(tmp_path):
    scenario(tmp_path, "a", "attempt-0001", "failed")
    scenario(tmp_path, "a", "attempt-0002", "complete", first=3_000_000_000)
    scenario(tmp_path, "b", "attempt-0001", "complete", "replay", 2_000_000_000)

    rows = p.write(tmp_path)

    assert [(row["scenario_id"], row["method"], row["migration_ttft_s"])
            for row in rows] == [("a", "kv_transfer", 2.0), ("b", "replay", 1.0)]
    assert (tmp_path / "migration_ttft_cdf.png").is_file()


def test_scenario_without_a_complete_attempt_fails(tmp_path):
    scenario(tmp_path, "a", "attempt-0001", "failed")

    try:
        p.extract(tmp_path)
    except ValueError as error:
        assert "complete attempt" in str(error)
    else:
        raise AssertionError("expected a failure for an all-failed scenario")
