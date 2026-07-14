"""Tests for the offline 10k-session experiment mechanics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import power_drain_experiment as e
from node_knee import node_loads


def test_a100_population_is_active_and_compute_packed():
    pool, pop = e.a100_population("agentic_tool_loop", 100, 0)

    assert len(pop) == 100
    assert set(pop.state) == {"active"}
    assert node_loads(pop).max() <= pool.rho_star + 1e-9


def test_completion_times_gives_each_source_an_independent_link():
    class Pop:
        source_node = np.array([0, 0, 1])

        def __len__(self):
            return 3

    imp = SimpleNamespace(
        c_replay=np.array([2.0, 2.0, 2.0]),
        c_transfer=np.array([3.0, 3.0, 3.0]),
        b_replay=np.array([125_000_000.0] * 3),
        b_transfer=np.array([250_000_000.0] * 3),
    )

    done = e.completion_times(Pop(), imp, np.array([0, 1, 2]), 1000, 40)

    assert np.allclose(done, [2.04, 3.04, 2.04])


def test_small_sweep_writes_canonical_outputs(tmp_path: Path):
    rows = e.run(100, target_fracs=(0.5,), bandwidths=(1000.0,), rtts=(40.0,))
    e.write(rows, tmp_path)

    assert len(rows) == len(e.WORKLOADS) * len(e.POLICIES)
    assert all(r["selected_w"] >= r["target_w"] for r in rows)
    assert (tmp_path / "scale_results.csv").exists()
    assert (tmp_path / "scale_policy_comparison.png").exists()
    assert (tmp_path / "scale_network_sensitivity.png").exists()
