import pytest
from shared_load_controls import actions


def test_matched_arm_counts():
    assert actions("none") == []
    assert actions("replay").count("replay") == 8
    assert actions("kv").count("kv_transfer") == 8
    assert actions("mixed").count("replay") == actions("mixed").count("kv_transfer") == 4
    with pytest.raises(ValueError): actions("mixed", 1)


@pytest.mark.parametrize('arms', [(), ('none', 'none'), ('unknown',)])
def test_invalid_selection_fails_before_hardware(monkeypatch, arms):
    from types import SimpleNamespace
    import shared_load_controls as controls
    monkeypatch.setattr(controls.network, 'configure_handoff_environment', lambda _: pytest.fail('hardware preparation reached'))
    with pytest.raises(ValueError, match='invalid matched control arms'):
        controls.run('openai/gpt-oss-20b', SimpleNamespace(destinations=(object(),)), {}, None, None, .05, arms=arms)
