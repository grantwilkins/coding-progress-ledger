import pytest

from agent_migrate_agent import segment_hash


def test_deterministic():
    assert segment_hash("repo context v1") == segment_hash("repo context v1")


def test_different_inputs_differ():
    assert segment_hash("a") != segment_hash("b")


def test_format_prefix():
    h = segment_hash("anything")
    assert h.startswith("h_") and len(h) == 18


def test_empty_rejected():
    with pytest.raises(ValueError):
        segment_hash("")


def test_non_str_rejected():
    with pytest.raises(TypeError):
        segment_hash(b"bytes_not_allowed")
