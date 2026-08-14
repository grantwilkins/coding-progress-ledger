"""Checks for the paired live repair/no-repair figure data reduction."""

import plot_repair_control_comparison as comparison


def _result(*, episode_id: str, requests: list[dict], target_s: float) -> dict:
    return {
        "episode_id": episode_id,
        "initial_moves": [{"session_id": "a"}, {"session_id": "b"}],
        "requests": requests,
        "time_to_target_s": target_s,
        "realized_shed_w": 31.0,
        "event_s": 5.0,
        "decision_s": 6.0,
        "requested_shed_w": 30.0,
    }


def _request(session_id: str, destination: str, method: str,
             ttft_s: float) -> dict:
    return {
        "session_id": session_id,
        "destination_instance": destination,
        "method": method,
        "ttft_s": ttft_s,
    }


def test_action_mix_accounts_for_every_original_session():
    result = _result(
        episode_id="repair",
        requests=[_request("a", "east", "replay", 2.0)],
        target_s=8.0,
    )

    assert comparison.action_mix(result) == {
        "east_replay": 1,
        "east_kv_transfer": 0,
        "germany_replay": 0,
        "germany_kv_transfer": 0,
        "not_moved": 1,
    }


def test_summary_preserves_the_paired_action_change():
    repaired = _result(
        episode_id="repair",
        requests=[_request("a", "east", "replay", 2.0)],
        target_s=8.0,
    )
    control = _result(
        episode_id="control",
        requests=[
            _request("a", "east", "replay", 2.0),
            _request("b", "germany", "kv_transfer", 20.0),
        ],
        target_s=12.0,
    )
    pairs = [{"repaired": repaired, "control": control} for _ in range(3)]

    summary = comparison.comparison_summary(pairs)

    assert summary["repeats"] == 3
    assert summary["action_mix"]["replan"]["not_moved"] == 1
    assert summary["action_mix"]["no_replan"]["germany_kv_transfer"] == 1
    assert summary["metrics"]["replan"]["means"]["time_to_target_s"] == 8.0
    assert summary["metrics"]["no_replan"]["means"]["ttft_max_s"] == 20.0
