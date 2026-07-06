from __future__ import annotations

"""
Claim:
Stage 1a reports node load as ell = f/F + g/G using the constants emitted
with the artifact.

Plausible wrong implementations:
- Reuse a stale saved ell array from an upstream fit.
- Drop either the prefill or decode term.
- Silently continue when raw f/g rate windows are missing.
- Plot a curve whose knee no longer matches the fitted power-knee definition.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

import stage1_profile as sp


def test_binned_curve_reports_average_power_by_ell_load():
    rows = sp.binned_curve([0.1, 0.2, 0.9], [10.0, 30.0, 90.0], 2)

    assert [r["n"] for r in rows] == [2, 1]
    assert rows[0]["ell_mean"] == pytest.approx(0.15)
    assert rows[0]["power_mean_w"] == pytest.approx(20.0)
    assert rows[1]["power_mean_w"] == pytest.approx(90.0)


def test_load_windows_recomputes_ell_from_rates(tmp_path: Path):
    path = tmp_path / "windows.npz"
    np.savez(
        path,
        P=np.array([10.0, 20.0]),
        f=np.array([2.0, 0.0]),
        g=np.array([0.0, 6.0]),
        ell=np.array([99.0, 99.0]),
    )

    windows = sp.load_windows(path, {"F_prefill_tps": 4.0, "G_decode_tps": 3.0})

    assert windows["ell"] == pytest.approx([0.5, 2.0])


def test_load_windows_hard_fails_when_rates_missing(tmp_path: Path):
    path = tmp_path / "windows.npz"
    np.savez(path, P=np.array([10.0]), ell=np.array([0.1]))

    with pytest.raises(ValueError, match="missing window fields: f, g"):
        sp.load_windows(path, {"F_prefill_tps": 4.0, "G_decode_tps": 3.0})


def test_concave_power_curve_hits_declared_power_knee():
    constants = {"P0_w": 100.0, "P_max_w": 300.0, "ell_power_knee": 2.0}
    rows = sp.concave_power_curve(constants, max_ell=2.0, points=2)

    assert rows[0]["power_w"] == pytest.approx(100.0)
    assert rows[-1]["ell"] == pytest.approx(2.0)
    assert rows[-1]["power_w"] == pytest.approx(260.0)


def test_read_constants_hard_fails_when_config_missing(tmp_path: Path):
    path = tmp_path / "summary.csv"
    fields = ("config", *sp.CONSTANT_FIELDS)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"config": "other", **{k: "1.0" for k in sp.CONSTANT_FIELDS}})

    with pytest.raises(ValueError, match="missing constants"):
        sp.read_constants(path, sp.CONFIG)
