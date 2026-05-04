import pytest
from pathlib import Path

from logrotator import rotate


def test_small_file_no_rotation(tmp_path):
    f = tmp_path / "app.log"
    f.write_bytes(b"hello world")
    result = rotate(str(f), 100)
    assert result == []
    assert f.exists()
    assert f.read_bytes() == b"hello world"


def test_empty_file_no_rotation(tmp_path):
    f = tmp_path / "app.log"
    f.write_bytes(b"")
    result = rotate(str(f), 10)
    assert result == []
    assert f.exists()


def test_exact_threshold_no_rotation(tmp_path):
    data = b"x" * 50
    f = tmp_path / "app.log"
    f.write_bytes(data)
    result = rotate(str(f), 50)
    assert result == []
    assert f.exists()
    assert f.read_bytes() == data


def test_large_file_splits_correctly(tmp_path):
    data = b"a" * (100 * 1024)
    f = tmp_path / "app.log"
    f.write_bytes(data)
    max_bytes = 30 * 1024
    result = rotate(str(f), max_bytes)
    assert len(result) == 4
    assert not f.exists()


def test_large_file_concat_equals_original(tmp_path):
    import os
    data = os.urandom(100 * 1024)
    f = tmp_path / "app.log"
    f.write_bytes(data)
    max_bytes = 30 * 1024
    result = rotate(str(f), max_bytes)
    recovered = b"".join(Path(p).read_bytes() for p in result)
    assert recovered == data


def test_parts_in_lexical_order(tmp_path):
    data = b"b" * (10 * 1024)
    f = tmp_path / "app.log"
    f.write_bytes(data)
    result = rotate(str(f), 3 * 1024)
    assert result == sorted(result)


def test_part_naming_four_digit_zero_padded(tmp_path):
    data = b"c" * 200
    f = tmp_path / "my.log"
    f.write_bytes(data)
    result = rotate(str(f), 90)
    assert result[0].endswith(".0001")
    assert result[1].endswith(".0002")
    assert result[2].endswith(".0003")


def test_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rotate(str(tmp_path / "nonexistent.log"), 1024)


def test_chunk_sizes_correct(tmp_path):
    data = b"d" * 100
    f = tmp_path / "app.log"
    f.write_bytes(data)
    result = rotate(str(f), 30)
    sizes = [Path(p).stat().st_size for p in result]
    assert sizes == [30, 30, 30, 10]
