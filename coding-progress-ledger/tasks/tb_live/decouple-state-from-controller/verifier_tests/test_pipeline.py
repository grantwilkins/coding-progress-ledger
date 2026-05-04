import copy
import pytest
from pipeline import process


def test_empty_input():
    assert process([], {}) == []


def test_all_rejected_by_filter():
    items = [{"status": "err", "name": "x"}, {"status": "fail", "name": "y"}]
    result = process(items, {"accept_status": "ok"})
    assert result == []


def test_filter_default_accept_status():
    items = [{"status": "ok", "val": 1}, {"status": "bad", "val": 2}]
    result = process(items, {})
    assert len(result) == 1
    assert result[0]["val"] == 1


def test_project_keeps_only_specified_fields():
    items = [{"status": "ok", "a": 1, "b": 2, "c": 3}]
    result = process(items, {"fields": ["a", "c"]})
    assert "a" in result[0]
    assert "c" in result[0]
    assert "b" not in result[0]


def test_project_missing_field_silently_omitted():
    items = [{"status": "ok", "a": 1}]
    result = process(items, {"fields": ["a", "nonexistent"]})
    assert "a" in result[0]
    assert "nonexistent" not in result[0]


def test_normalize_lowercases_and_strips():
    items = [{"status": "ok", "name": "  Alice  ", "city": " NYC "}]
    result = process(items, {})
    assert result[0]["name"] == "alice"
    assert result[0]["city"] == "nyc"


def test_normalize_leaves_non_strings_unchanged():
    items = [{"status": "ok", "score": 42, "active": True}]
    result = process(items, {})
    assert result[0]["score"] == 42
    assert result[0]["active"] is True


def test_tag_adds_pipeline_version_default():
    items = [{"status": "ok"}]
    result = process(items, {})
    assert result[0]["_pipeline_version"] == "v1"


def test_tag_uses_config_version():
    items = [{"status": "ok"}]
    result = process(items, {"version": "v42"})
    assert result[0]["_pipeline_version"] == "v42"


def test_sort_ascending_by_key():
    items = [
        {"status": "ok", "score": 3},
        {"status": "ok", "score": 1},
        {"status": "ok", "score": 2},
    ]
    result = process(items, {"sort_key": "score"})
    assert [r["score"] for r in result] == [1, 2, 3]


def test_sort_stable_on_tied_key():
    items = [
        {"status": "ok", "rank": 1, "id": "a"},
        {"status": "ok", "rank": 1, "id": "b"},
        {"status": "ok", "rank": 1, "id": "c"},
    ]
    result = process(items, {"sort_key": "rank"})
    assert [r["id"] for r in result] == ["a", "b", "c"]


def test_no_sort_key_preserves_order():
    items = [
        {"status": "ok", "id": "z"},
        {"status": "ok", "id": "a"},
        {"status": "ok", "id": "m"},
    ]
    result = process(items, {})
    assert [r["id"] for r in result] == ["z", "a", "m"]


def test_limit_truncates():
    items = [{"status": "ok", "n": i} for i in range(5)]
    result = process(items, {"limit": 2})
    assert len(result) == 2
    assert result[0]["n"] == 0
    assert result[1]["n"] == 1


def test_limit_default_no_truncation():
    items = [{"status": "ok", "n": i} for i in range(4)]
    result = process(items, {})
    assert len(result) == 4


def test_pure_input_list_unmodified():
    items = [{"status": "ok", "name": "  Bob  "}]
    original = copy.deepcopy(items)
    process(items, {"version": "v2"})
    assert items == original


def test_pure_input_dicts_unmodified():
    d = {"status": "ok", "name": "  Carol  ", "score": 5}
    items = [d]
    process(items, {})
    assert d == {"status": "ok", "name": "  Carol  ", "score": 5}


def test_end_to_end_all_stages():
    items = [
        {"status": "ok",  "name": "  Alice ", "score": 3, "extra": "drop"},
        {"status": "err", "name": "Bob",      "score": 1, "extra": "drop"},
        {"status": "ok",  "name": " Carol",   "score": 1, "extra": "drop"},
        {"status": "ok",  "name": " Dave ",   "score": 2, "extra": "drop"},
    ]
    config = {
        "accept_status": "ok",
        "fields": ["name", "score"],
        "version": "v2",
        "sort_key": "score",
        "limit": 2,
    }
    result = process(items, config)
    assert len(result) == 2
    assert result[0] == {"name": "carol", "score": 1, "_pipeline_version": "v2"}
    assert result[1] == {"name": "dave", "score": 2, "_pipeline_version": "v2"}
    # extra key dropped, err item absent, alice (score=3) beyond limit
