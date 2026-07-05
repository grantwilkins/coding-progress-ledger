"""Claim:
The kappa sweep replays kappa-derated MILP plans through the DES on the shared
full-node-expected target basis.

Plausible wrong implementations:
- Report selected (planned) relief as realized relief.
- Compute miss fraction over all jobs instead of moved sessions.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_node_knee_kappa_sweep import run_sweep


def test_kappa_sweep_realized_bounded_by_selected():
    _, rows = run_sweep(kappas=(0.7, 1.0), deadlines=(10.0,))
    assert len(rows) == 2
    for r in rows:
        assert r["target_basis"] == "full_node_expected"
        assert r["rebuild_node_kw"] <= r["selected_node_kw"] + 1e-9  # realized ⊆ selected
        assert 0 < r["movers"] <= r["jobs"]
        # miss_frac is per MOVED session: (misses / movers) lands on a 1/movers grid
        assert (r["miss_frac"] * r["movers"]) == pytest.approx(round(r["miss_frac"] * r["movers"]))
