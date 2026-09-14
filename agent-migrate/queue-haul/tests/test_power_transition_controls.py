import pytest
import power_transition_controls as controls
import migration_testbed as testbed


def test_recorded_schedule_preserves_interturn_gaps_and_shapes():
    manifest = {"sessions": [{"id": "s", "turns": [
        {"time_s": 100, "input_tokens": 1000, "output_tokens": 30},
        {"time_s": 103, "input_tokens": 1100, "output_tokens": 50}]}]}
    rows = controls.schedules(manifest, "coding_trace", 10, 0)
    assert [row[0] for row in rows] == [0, 3, 6, 9]
    assert [row[2:4] for row in rows] == [(1000, 30), (1100, 50)] * 2
    manifest["sessions"][0]["turns"][0]["input_tokens"] = 32768
    with pytest.raises(ValueError, match="does not fit"):
        controls.schedules(manifest, "coding_trace", 10, 0)


@pytest.mark.parametrize("model", testbed.MODEL_SPECS)
def test_serving_runtime_preserves_geometry_and_enables_compilation(monkeypatch, model):
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    cfg = testbed.model_campaign_config(model, serving=True)
    testbed.validate_model_runtime(cfg)
    assert cfg.max_num_seqs == 256
    assert cfg.max_num_batched_tokens == testbed.model_spec(model).batched_tokens
    assert not cfg.enforce_eager
    assert not cfg.literal_token_timing
    assert testbed.model_chunk_tokens(cfg) == testbed.model_spec(model).chunk_tokens
    assert testbed.model_campaign_config(model).max_num_seqs == 8
