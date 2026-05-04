from pathlib import Path

import pytest

from conftest import CORRUPT_DB, INTACT_DB, MIN_SURVIVING_ROWS, NUM_ROWS, ORIGINAL_ROWS


def _orig(row_id: int) -> dict:
    t = ORIGINAL_ROWS[row_id - 1]
    return {"id": t[0], "name": t[1], "value": t[2]}


def test_intact_returns_all_rows():
    from sqlite_recover import recover
    rows = recover(str(INTACT_DB))
    assert len(rows) == NUM_ROWS


def test_intact_rows_in_id_order():
    from sqlite_recover import recover
    rows = recover(str(INTACT_DB))
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)


def test_intact_rows_match_originals():
    from sqlite_recover import recover
    rows = recover(str(INTACT_DB))
    for row in rows:
        assert row == _orig(row["id"])


def test_truncated_returns_at_least_min_rows():
    from sqlite_recover import recover
    rows = recover(str(CORRUPT_DB))
    assert len(rows) >= MIN_SURVIVING_ROWS


def test_truncated_rows_match_originals():
    from sqlite_recover import recover
    rows = recover(str(CORRUPT_DB))
    for row in rows:
        assert row == _orig(row["id"])


def test_truncated_rows_in_id_order():
    from sqlite_recover import recover
    rows = recover(str(CORRUPT_DB))
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids)


def test_missing_file_raises():
    from sqlite_recover import recover
    with pytest.raises(FileNotFoundError):
        recover("/nonexistent/path/no.sqlite")


def test_empty_file_returns_empty(tmp_path):
    from sqlite_recover import recover
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(b"")
    assert recover(str(empty)) == []
