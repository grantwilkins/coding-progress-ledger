import json

import matched_action_campaign as campaign


def test_matched_action_campaign_is_distinct_and_resumable(tmp_path):
    summary = campaign.run(tmp_path)
    assert summary["status"] == "complete"
    assert all(summary["gates"].values())
    assert summary["completed_a100_run"] == {
        "status": "complete", "deadline_met": True, "target_met": True,
        "migration_s": 16.459174637,
        "requested_shed_w": 49.48530596349974,
        "realized_shed_w": 61.856632454374676,
        "request_failures": 0,
        "action_counts": {**dict.fromkeys(campaign.ACTIONS, 0),
                          "germany_kv_transfer": 8},
    }
    arms = {row["arm_id"]: row for row in summary["arms"]}
    assert arms["gpt_oss_20b_a100"]["method_counts"] == {
        "replay": 0, "kv_transfer": 8}
    assert arms["gpt_oss_20b_h100"]["method_counts"] == {
        "replay": 6, "kv_transfer": 0}
    assert arms["qwen3_8_27b_h100"]["method_counts"] == {
        "replay": 0, "kv_transfer": 7}
    assert arms["gemma_4_26b_h100"]["method_counts"] == {
        "replay": 6, "kv_transfer": 0}
    assert not arms["gemma_4_26b_h100"]["feasible"]
    checkpoints = sorted((tmp_path / "arms").glob("*.json"))
    before = {path: path.stat().st_mtime_ns for path in checkpoints}
    assert campaign.run(tmp_path)["input_sha256"] == summary["input_sha256"]
    assert {path: path.stat().st_mtime_ns for path in checkpoints} == before
    assert json.loads((tmp_path / "summary.json").read_text())["gates"] == summary["gates"]
    for suffix in ("csv", "png", "pdf"):
        assert (tmp_path / f"action_mix.{suffix}").is_file()
    assert (tmp_path / "bootstrap_action_mix.csv").is_file()
    assert all(row["samples"] == 200
               for row in summary["power_bootstrap"].values())
