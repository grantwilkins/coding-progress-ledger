from __future__ import annotations

import csv
from pathlib import Path

import pytest

import stage1_profile as sp


def test_binned_curve_reports_average_power_by_ell_load():
    rows = sp.binned_curve([0.1, 0.2, 0.9], [10.0, 30.0, 90.0], 2)

    assert [r["n"] for r in rows] == [2, 1]
    assert rows[0]["ell_mean"] == pytest.approx(0.15)
    assert rows[0]["power_mean_w"] == pytest.approx(20.0)
    assert rows[1]["power_mean_w"] == pytest.approx(90.0)


def test_read_constants_hard_fails_when_config_missing(tmp_path: Path):
    path = tmp_path / "summary.csv"
    fields = ("config", *sp.CONSTANT_FIELDS)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"config": "other", **{k: "1.0" for k in sp.CONSTANT_FIELDS}})

    with pytest.raises(ValueError, match="missing constants"):
        sp.read_constants(path, sp.CONFIG)
