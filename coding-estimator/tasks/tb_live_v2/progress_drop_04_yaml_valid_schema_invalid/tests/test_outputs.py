"""Verifier for progress_drop_04_yaml_valid_schema_invalid."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
CFG = WS / "config.yaml"


def _load() -> dict:
    return yaml.safe_load(CFG.read_text())


def test_file_exists():
    assert CFG.is_file(), f"{CFG} not present"


def test_top_level_keys_present():
    d = _load()
    for k in ("service", "replicas", "port", "env", "resources"):
        assert k in d, f"missing top-level key: {k}"


def test_service_is_nonempty_string():
    d = _load()
    assert isinstance(d["service"], str) and d["service"], "service must be non-empty string"


def test_replicas_is_int_ge_1():
    d = _load()
    assert isinstance(d["replicas"], int) and not isinstance(d["replicas"], bool)
    assert d["replicas"] >= 1


def test_port_is_int_in_range():
    d = _load()
    assert isinstance(d["port"], int) and not isinstance(d["port"], bool)
    assert 1 <= d["port"] <= 65535


def test_env_is_str_str_mapping():
    d = _load()
    assert isinstance(d["env"], dict)
    for k, v in d["env"].items():
        assert isinstance(k, str) and isinstance(v, str)


def test_resources_has_required_subkeys():
    d = _load()
    r = d["resources"]
    assert isinstance(r, dict)
    assert "cpu" in r and isinstance(r["cpu"], str) and r["cpu"]
    assert "memory" in r and isinstance(r["memory"], str) and r["memory"]
