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


def test_plot_uses_actual_ell_and_normalizes_power(monkeypatch, tmp_path: Path):
    figures = []
    monkeypatch.setattr(sp.plt, "close", figures.append)
    sp.make_plot(
        {"ell": np.array([0.25, 1.0]), "P": np.array([100.0, 400.0])},
        [{"ell": 0.0, "power_w": 0.0}, {"ell": 1.0, "power_w": 400.0}],
        tmp_path / "plot",
    )

    ax = figures[0].axes[0]
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(), [[0.25, 0.25], [1.0, 1.0]]
    )
    assert ax.collections[0].get_alpha() == pytest.approx(0.1)
    assert ax.lines[0].get_ydata() == pytest.approx([0.0, 1.0])
    assert ax.lines[0].get_label() == "Power Curve"
    assert ax.get_xlabel() == r"Load ($\ell$)"
    assert not ax.get_title()


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
