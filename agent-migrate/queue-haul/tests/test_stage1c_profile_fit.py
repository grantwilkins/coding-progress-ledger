"""
Claim:
Stage 1C calibration fits repeats 0 and 1, evaluates repeat 2, and models KV
network transfer and destination ingestion as overlapping work.

Plausible wrong implementations:
- Fit the held-out repeat and report training error as validation.
- Add network and destination KV times.
- Fit replay from requested context instead of measured processed tokens.
- Reverse bytes/s and seconds/byte.
- Resend the changed partial block instead of replaying its tail.
- Append measured and prior curve points out of order.
"""

import pandas as pd
import pytest

import stage1c_profile_fit as fit


def test_fit_uses_only_training_repeats_and_pipeline_time():
    replay = pd.DataFrame([
        {"method": "replay", "concurrency": 1, "activity": "none", "repeat": repeat,
         "measured_prompt_tokens": prompt, "measured_processed_tokens": prompt,
         "initial_time_to_first_response_s": prompt / (50 if repeat == 2 else 100),
         "bandwidth_mbps": 1000}
        for repeat in range(3) for prompt in (10, 20)
    ])
    kv = pd.DataFrame([
        {"method": "kv_transfer", "concurrency": 1, "activity": "none",
         "repeat": repeat, "measured_prompt_tokens": size,
         "measured_kv_bytes": size, "initial_time_to_first_response_s": size / 50,
         "bandwidth_mbps": 10_000}
        for repeat in range(3) for size in (100, 200)
    ])
    held_out = {**kv.iloc[0].to_dict(), "repeat": 2, "bandwidth_mbps": .0002,
                "initial_time_to_first_response_s": 4}
    rows = pd.concat([replay, kv, pd.DataFrame([held_out])], ignore_index=True)

    curve, replay_error = fit.fit_replay(rows)
    destination_rate, kv_error = fit.fit_kv(rows)

    assert curve == [[10.0, 100.0], [20.0, 100.0]]
    assert replay_error == pytest.approx(.5)
    assert destination_rate == pytest.approx(50)
    assert kv_error == pytest.approx(0)


def test_catch_up_fit_separates_completed_blocks_tail_and_fixed_completion():
    rows = pd.DataFrame([
        {
            "method": "kv_transfer", "repeat": repeat,
            "measured_prompt_tokens": 295, "catch_up_prompt_tokens": final,
            "bandwidth_mbps": .0008,
            "catch_up_time_to_first_response_s": network + 1.02,
            "catch_up_response_s": .5, "catch_up_validation_s": .2,
        }
        for repeat in (0, 1)
        for final, network in ((358, 0), (614, 1))
    ])

    tail_tps, fixed_s = fit.fit_catch_up(rows, 256, 100)

    assert tail_tps == pytest.approx(100)
    assert fixed_s == pytest.approx(.7)


def test_measured_replay_points_override_priors_and_remain_ordered():
    assert fit.merge_points(
        [[20, 2], [10, 1]], [[15, 9], [20, 99]]
    ) == [[10.0, 1.0], [15.0, 9.0], [20.0, 2.0]]
