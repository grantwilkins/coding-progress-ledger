import json

import plot_network_ttft_cdf as p


def scenario(root, name, attempt, status, method="kv_transfer", first=3_000_000_000):
    path = root / "scenarios" / name / attempt
    path.mkdir(parents=True)
    body = {"scenario_id": name, "status": status}
    if status == "complete":
        body |= {"started_ns": 1_000_000_000, "ended_ns": 4_000_000_000,
                 "migration_s": 3.0, "requests": [{
                     "method": method, "order": 0, "session_id": f"s:{name}",
                     "destination_instance": "west", "request": {
                         "start_ns": 1_000_000_000, "first_byte_ns": first,
                         "end_ns": 4_000_000_000}}]}
    (path / "result.json").write_text(json.dumps(body))
    (path / "scenario.json").write_text(json.dumps({
        "bandwidth": "controlled_80", "bandwidth_mbps": {"west": 7400},
        "policy": "queue_haul", "workload": "agentic_tool_loop"}))


def test_retried_attempt_supersedes_preempted_failure(tmp_path):
    scenario(tmp_path, "a", "attempt-0001", "failed")
    scenario(tmp_path, "a", "attempt-0002", "complete", first=3_000_000_000)
    scenario(tmp_path, "b", "attempt-0001", "complete", "replay", 2_000_000_000)

    rows = p.write(tmp_path)

    assert [(row["scenario_id"], row["method"], row["migration_ttft_s"],
             row["bandwidth_mbps"]) for row in rows] == [
                 ("a", "kv_transfer", 2.0, 7400), ("b", "replay", 1.0, 7400)]
    assert (tmp_path / "migration_ttft_cdf.png").is_file()


def test_partial_campaign_reports_skipped_scenarios(tmp_path, capsys):
    scenario(tmp_path, "a", "attempt-0001", "failed")
    scenario(tmp_path, "b", "attempt-0001", "complete")

    rows = p.extract(tmp_path)

    assert [row["scenario_id"] for row in rows] == ["b"]
    assert "skipping 1 scenarios" in capsys.readouterr().out
