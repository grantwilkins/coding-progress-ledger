"""Path resolver tests, exercised against a fixture ledger_root and the real one."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_estimator.ingest import paths


def _build_fixture(tmp_path: Path) -> Path:
    (tmp_path / "ledger_progress").mkdir()
    runs = tmp_path / "runs" / "tb_live"
    runs.mkdir(parents=True)
    (runs / "task_b").mkdir()
    (runs / "task_a").mkdir()
    (runs / "not_a_dir").write_text("file")
    return tmp_path


def test_ledger_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(fixture))
    assert paths.ledger_root() == fixture.resolve()


def test_ledger_root_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEDGER_ROOT", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        paths.ledger_root()


def test_ledger_root_not_a_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEDGER_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        paths.ledger_root()


def test_runs_root_unknown_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(fixture))
    with pytest.raises(KeyError):
        paths.runs_root("does_not_exist")


def test_list_run_ids_sorted_and_dirs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(fixture))
    ids = paths.list_run_ids("tb_live")
    assert ids == ["task_a", "task_b"]


def test_run_dir_returns_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(fixture))
    path = paths.run_dir("tb_live", "task_a")
    assert path.name == "task_a"
    with pytest.raises(FileNotFoundError):
        paths.run_dir("tb_live", "nope")


def test_real_ledger_root_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEDGER_ROOT", raising=False)
    real = paths.ledger_root()
    assert (real / "ledger_progress").is_dir()
    # tb_live is the only first-party live source we ship; its presence is a
    # strong signal that the upstream checkout is healthy.
    assert (real / "runs" / "tb_live").is_dir()
    ids = paths.list_run_ids("tb_live")
    assert "markdown-to-html-cli" in ids
    assert ids == sorted(ids)
    assert os.path.isabs(str(real))
