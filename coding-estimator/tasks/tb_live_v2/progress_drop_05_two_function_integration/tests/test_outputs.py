"""Verifier for progress_drop_05_two_function_integration."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

WS = Path(os.environ.get("TB_LIVE_V2_WORKSPACE", str(Path.cwd())))
APP = WS / "pipeline.py"


def _csv(tmp_path: Path) -> Path:
    p = tmp_path / "orders.csv"
    p.write_text(textwrap.dedent("""\
        id,qty,unit_price
        a,3,2.5
        b,2,4.0
        c,1,1.25
    """))
    return p


def test_module_exists():
    assert APP.is_file(), f"{APP} not present"


def test_cli_total_revenue(tmp_path):
    p = _csv(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(APP), str(p)],
        capture_output=True, text=True, timeout=10, check=True,
    )
    # 3*2.5 + 2*4.0 + 1*1.25 = 16.75
    assert proc.stdout.strip() == "total_revenue=16.75"


def test_total_revenue_is_float():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.total_revenue([{"id": "a", "qty": 2, "unit_price": 1.5}])
    assert isinstance(out, float)
    assert out == 3.0


def test_parse_orders_returns_list_of_dicts(tmp_path):
    import importlib.util
    p = _csv(tmp_path)
    spec = importlib.util.spec_from_file_location("pipeline", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.parse_orders(str(p))
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(r, dict) for r in out)
    assert {"id", "qty", "unit_price"}.issubset(out[0].keys())


def test_integration_handoff_works(tmp_path):
    """parse_orders → total_revenue must compose."""
    import importlib.util
    p = _csv(tmp_path)
    spec = importlib.util.spec_from_file_location("pipeline", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.parse_orders(str(p))
    assert mod.total_revenue(rows) == 16.75
