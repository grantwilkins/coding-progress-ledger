"""Native transport accounting and fixed-split regional component evidence."""

import numpy as np
import pytest

from pool_shed_calibration import CROSSOVER, LONG_PACKS, _read, calibration, kv_state, loaded_execution_check


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
