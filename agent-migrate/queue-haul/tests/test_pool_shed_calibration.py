"""Native transport accounting and fixed-split regional component evidence."""

import numpy as np
import pytest

from pool_shed_calibration import CROSSOVER, LONG_PACKS, _read, calibration, kv_state, loaded_execution_check, resident_execution_check, service_work


def test_source_power_uses_declared_rates_and_per_draw_idle_without_extrapolation():
    from dataclasses import replace
    from pool_shed_campaign import sample_fleet
    from pool_shed_calibration import POWER_PROFILE, source_power
    from profiles import ModelProfile

    fleet, measured = sample_fleet("coding"), calibration(0)
    actual, curve = source_power(fleet, measured), ModelProfile.load(POWER_PROFILE).case().phase_power
    assert actual["active_w"] < measured["active_w"]
    assert actual["delta_w"] == pytest.approx(actual["active_w"] - actual["idle_w"])
    np.testing.assert_allclose(actual["delta_draws_w"], [np.interp(actual["power_load"], *np.asarray(c).T) - c[0][1]
                                                       for c in curve.measured_power_bootstrap])
    idle = replace(fleet, metadata={**fleet.metadata, "source_session_rps": 0.})
    assert source_power(idle, measured)["delta_w"] == 0
    assert not np.any(source_power(idle, measured)["delta_draws_w"])
    overloaded = replace(fleet, metadata={**fleet.metadata, "source_session_rps": 1e9})
    with pytest.raises(ValueError, match="power hull"):
        source_power(overloaded, measured)


def test_resident_service_uses_matched_context_anchors_and_explicit_evidence_scope():
    value = calibration(0)
    service = value["resident_service"]
    expected = (2048 / np.array([5550.823048297525, 4376.4708558464745, 5337.344115334956])
                + 32 / np.array([1109.6084258150277, 317.97384033307566, 174.8848660847107])) / .1140625
    np.testing.assert_allclose(service_work([4096, 16384, 24576], 2048, 32, value), expected)
    assert service_work(4096, 2048, 32, value) == pytest.approx(expected[0])
    assert len(service["evidence"]) == 4 and service["full_profile_accepted"] is False
    assert service["context_limit"] == 31562 and service["sources"].items() <= value["sources"].items()
    assert service_work(100000, 2048, 32, value) >= expected.max()
    for context, prompt, output in ((0, 1, 1), (4096, -1, 1), (4096, 1, np.nan)):
        with pytest.raises(ValueError, match="resident request shape"):
            service_work(context, prompt, output, value)


def test_sealed_payload_matches_hardware_and_preserves_partial_tokens():
    value = calibration(0)
    rows = [r for path in (CROSSOVER, LONG_PACKS) for r in _read(path / "migrations.csv")
            if r["method"] == "kv_transfer"]
    contexts = np.array([int(r["measured_prompt_tokens"]) for r in rows])
    state, residual = kv_state(contexts, value)
    assert len(rows) == 1540
    np.testing.assert_array_equal(state, [int(r["measured_kv_bytes"]) for r in rows])
    np.testing.assert_array_equal(state // value["kv_block_bytes"] * value["kv_block_tokens"] + residual, contexts)
    assert np.any(residual > 0)
    np.testing.assert_array_equal(kv_state([0, 255, 256, 257], value),
                                  [[0, 0, value["kv_block_bytes"], value["kv_block_bytes"]], [0, 255, 0, 1]])
    for invalid in ([-1], [1.5], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="token counts"):
            kv_state(invalid, value)


def test_regional_primitive_fit_passes_preserved_gate_and_exposes_residuals():
    regional = calibration(0)["regional_components"]
    assert regional["training_episodes"] == 53 and regional["heldout_episodes"] == 24
    assert regional["gates"] == {"mae_s": 3, "r2": .8}
    assert regional["validation"]["gate_pass"]
    assert len(regional["validation"]["aggregate"]["residual_s"]) == 24
    assert regional["validation"]["aggregate"]["false_feasible_25s"] > 0
    assert np.all(np.array(regional["endpoint_bytes_per_s"]) * 8 < [2.2802e9, 8.7334e9])


def test_independent_engine_reproduces_original_loaded_replay_validation():
    report = loaded_execution_check(calibration(0))
    assert report["episodes"] == 220 and report["gate_pass"]
    assert report["false_feasible_25s"] == 0


def test_resident_replay_loss_uses_paired_training_and_preserves_kv_negative_control():
    value = calibration(2)
    report = value["resident_interference"]
    assert report["training_routes"] == 31 and report["training_episodes"] == 23
    assert .85 < report["replay_loss"] < .95
    assert report["validation"]["fixed_replay"]["routes"] == 6
    assert report["validation"]["fixed_replay"]["mae_requests"] < 1
    assert report["validation"]["fixed_kv_transfer"]["median_observed_loss"] < .05
    assert {r["load"] for r in report["measurements"]} == {.25, .5}
    train = {r["scenario_id"] for r in report["measurements"] if r["training"]}
    heldout = {r["scenario_id"] for r in report["measurements"] if not r["training"]}
    assert not train & heldout
    assert all(not heldout.intersection(ids) for ids in value["regional_components"]["bootstrap_episode_ids"])
    assert value["timing"][0]["resident_replay_loss"] == report["replay_loss"]
    assert len({t["resident_replay_loss"] for t in value["timing"]}) > 1
    assert all("resident_kv_loss" not in t for t in value["timing"])


def test_independent_resident_debt_prediction_preserves_measured_windows():
    report = resident_execution_check(calibration(0))
    assert report["routes"] == 6 and report["episodes"] == 5
    assert report["normalized_mae"] < .1
    assert any(r["observed_migration_s"] != r["predicted_migration_s"] for r in report["predictions"])
    assert all(r["resident_debt_work_s"] > 0 for r in report["predictions"])
