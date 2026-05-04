"""Verifier tests for safetar.safe_extract."""
import io
import os
import tarfile

import pytest

from safetar import UnsafeTarError, safe_extract


# ---------------------------------------------------------------------------
# Helpers to build tars entirely in memory, written to tmp_path
# ---------------------------------------------------------------------------

def _make_tar(tmp_path, name, members):
    """
    members: list of (TarInfo, bytes|None) tuples.
    Returns path to the written tar file.
    """
    tar_path = tmp_path / name
    with tarfile.open(str(tar_path), "w") as tf:
        for info, data in members:
            if data is not None:
                tf.addfile(info, io.BytesIO(data))
            else:
                tf.addfile(info)
    return tar_path


def _file_member(name, content=b"hello"):
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    return info, content


def _symlink_member(name, target):
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _hardlink_member(name, target):
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info, None


# ---------------------------------------------------------------------------
# Benign tars
# ---------------------------------------------------------------------------

def test_benign_single_file(tmp_path):
    tar_path = _make_tar(tmp_path, "benign1.tar", [_file_member("hello.txt", b"world")])
    dest = tmp_path / "out1"
    dest.mkdir()
    safe_extract(str(tar_path), str(dest))
    assert (dest / "hello.txt").read_bytes() == b"world"


def test_benign_nested_paths(tmp_path):
    tar_path = _make_tar(tmp_path, "benign2.tar", [
        _file_member("a/b/c.txt", b"nested"),
        _file_member("a/d.txt", b"sibling"),
    ])
    dest = tmp_path / "out2"
    dest.mkdir()
    safe_extract(str(tar_path), str(dest))
    assert (dest / "a" / "b" / "c.txt").read_bytes() == b"nested"
    assert (dest / "a" / "d.txt").read_bytes() == b"sibling"


# ---------------------------------------------------------------------------
# Malicious: path traversal via ..
# ---------------------------------------------------------------------------

def test_path_traversal_raises(tmp_path):
    tar_path = _make_tar(tmp_path, "traversal.tar", [_file_member("../escape.txt")])
    dest = tmp_path / "out3"
    dest.mkdir()
    with pytest.raises(UnsafeTarError):
        safe_extract(str(tar_path), str(dest))


# ---------------------------------------------------------------------------
# Malicious: absolute path
# ---------------------------------------------------------------------------

def test_absolute_path_raises(tmp_path):
    info = tarfile.TarInfo(name="/etc/escape.txt")
    info.size = 5
    tar_path = _make_tar(tmp_path, "abspath.tar", [(info, b"oops!")])
    dest = tmp_path / "out4"
    dest.mkdir()
    with pytest.raises(UnsafeTarError):
        safe_extract(str(tar_path), str(dest))


# ---------------------------------------------------------------------------
# Malicious: symlink pointing outside dest_dir
# ---------------------------------------------------------------------------

def test_symlink_escape_raises(tmp_path):
    tar_path = _make_tar(tmp_path, "symlink.tar", [
        _symlink_member("link", "../../outside"),
    ])
    dest = tmp_path / "out5"
    dest.mkdir()
    with pytest.raises(UnsafeTarError):
        safe_extract(str(tar_path), str(dest))


# ---------------------------------------------------------------------------
# Malicious: hardlink targeting absolute path
# ---------------------------------------------------------------------------

def test_hardlink_escape_raises(tmp_path):
    tar_path = _make_tar(tmp_path, "hardlink.tar", [
        _hardlink_member("bad_link", "/etc/passwd"),
    ])
    dest = tmp_path / "out6"
    dest.mkdir()
    with pytest.raises(UnsafeTarError):
        safe_extract(str(tar_path), str(dest))


# ---------------------------------------------------------------------------
# Atomic: no partial extraction when malicious member detected
# ---------------------------------------------------------------------------

def test_no_partial_extraction(tmp_path):
    # First member is benign, second triggers traversal
    tar_path = _make_tar(tmp_path, "partial.tar", [
        _file_member("good.txt", b"safe"),
        _file_member("../escape.txt", b"evil"),
    ])
    dest = tmp_path / "out7"
    dest.mkdir()
    with pytest.raises(UnsafeTarError):
        safe_extract(str(tar_path), str(dest))
    # dest_dir must be empty — nothing was extracted
    assert list(dest.iterdir()) == []
