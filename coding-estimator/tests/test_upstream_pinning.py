"""Pinning manifest is deterministic at a fixed upstream SHA, fails fast on dirty."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from coding_estimator.ingest import pinning


def _init_fixture_ledger(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "upstream"
    root.mkdir()
    (root / "ledger_progress").mkdir()
    art_dir = root / "datasets"
    art_dir.mkdir()
    art = art_dir / "table.csv"
    art.write_text("a,b\n1,2\n", encoding="utf-8")
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=root)
    subprocess.check_call(["git", "add", "."], cwd=root)
    subprocess.check_call(["git", "commit", "-q", "-m", "init"], cwd=root)
    return root, art


def test_pin_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, art = _init_fixture_ledger(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(root))
    out1 = tmp_path / "pin1.json"
    out2 = tmp_path / "pin2.json"
    pinning.capture_pin({"w3_table": art}, out1)
    pinning.capture_pin({"w3_table": art}, out2)
    a = json.loads(out1.read_text())
    b = json.loads(out2.read_text())
    assert a["ledger_commit_sha"] == b["ledger_commit_sha"]
    assert a["artifacts"] == b["artifacts"]
    assert a["artifacts"]["w3_table"]["path"] == "datasets/table.csv"


def test_dirty_tree_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, art = _init_fixture_ledger(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(root))
    art.write_text("a,b\n9,9\n", encoding="utf-8")  # dirty
    with pytest.raises(RuntimeError, match="dirty"):
        pinning.capture_pin({"w3_table": art}, tmp_path / "pin.json")


def test_dirty_tree_allow_dirty_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, art = _init_fixture_ledger(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(root))
    art.write_text("a,b\n9,9\n", encoding="utf-8")
    out = pinning.capture_pin({"w3_table": art}, tmp_path / "pin.json", allow_dirty=True)
    assert json.loads(out.read_text())["ledger_dirty"] is True


def test_artifact_outside_ledger_root_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _init_fixture_ledger(tmp_path)
    monkeypatch.setenv("LEDGER_ROOT", str(root))
    outside = tmp_path / "outside.csv"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not under ledger_root"):
        pinning.capture_pin({"k": outside}, tmp_path / "pin.json")
