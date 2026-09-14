import pytest
from shared_load_controls import actions


def test_matched_arm_counts():
    assert actions("none") == []
    assert actions("replay").count("replay") == 8
    assert actions("kv").count("kv_transfer") == 8
    assert actions("mixed").count("replay") == actions("mixed").count("kv_transfer") == 4
    with pytest.raises(ValueError): actions("mixed", 1)
