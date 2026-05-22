"""
Claim:
The retained-state drain frontier reports the largest rounded queue-safe
retained-prefill fraction for each allocation policy, workload seed, and drain
window under EDF release, using absolute event-start deadlines while tying
solver capacity and reported drain budgets to the same x-value.

Plausible wrong implementations:
- Keep using release-relative reconstruction safety after adding drain-order
  ablations.
- Reuse the capacity horizon as a release delay and make larger windows miss
  absolute event-start deadlines.
- Keep the old monotone available-window envelope after switching to absolute
  event-start deadlines.
- Trust binary search monotonicity despite rounded nonmonotone safety.
- Collapse workload seeds before preserving per-seed frontier rows.
- Drop the least-loaded ordinary-routing baseline from the north-star plot.
- Drop EDF release from the frontier sweep.
- Forget to sample the fixed H1 stress target when frontier search lands elsewhere.
- Keep writing the old deadline-scale frontier CSV names.
- Mark rows safe when they exceed physical pressure or the drain window.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import experiments.run_retained_state_frontier as retained
from experiments.run_retained_state_frontier import (
    DRAIN_WINDOWS_S,
    FRONTIER_POLICIES,
    FRONTIER_RELEASE_POLICIES,
    MAIN_POLICY,
    QUEUE_RELEASE_SPAN_S,
    STRESS_FRACTIONS,
    WORKLOAD_SEEDS,
    _default_workload_seeds,
    _failure_mode,
    _frontier_job,
    _is_safe,
    _monotone_frontier,
    _policy_row,
    _with_window,
)
from evaluation import WorkloadConfig
from problem import ProblemData


def test_drain_windows_are_positive_log_plot_points():
    assert DRAIN_WINDOWS_S == (
        10.0,
        12.5,
        16.0,
        20.0,
        25.0,
        32.0,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
        600.0,
        800.0,
        1000.0,
        1200.0,
        1600.0,
        2000.0,
        2400.0,
        3000.0,
        3600.0,
    )
    assert all(window > 0.0 for window in DRAIN_WINDOWS_S)


def test_generated_frontier_uses_many_workload_seeds_for_error_bars():
    assert WORKLOAD_SEEDS == tuple(range(16))
    assert _default_workload_seeds(WorkloadConfig(source="generated")) == WORKLOAD_SEEDS
    assert _default_workload_seeds(WorkloadConfig(source="fixed", seed=3)) == (3,)
    assert QUEUE_RELEASE_SPAN_S == 0.0
    assert FRONTIER_RELEASE_POLICIES == ("edf",)


def test_frontier_policy_set_includes_least_loaded_baseline():
    assert MAIN_POLICY == "deadline-penalty-rounded"
    assert FRONTIER_POLICIES == (
        "deadline-penalty-rounded",
        "online-queue-greedy",
        "least-loaded-destination",
        "replay-only",
        "state-only",
    )


def test_window_rescale_reuses_workload_and_preserves_background_load_fraction():
    problem = ProblemData(
        model=SimpleNamespace(prefill_tok_s=10.0),
        regime="toy",
        T=np.array([100.0]),
        d=np.array([2.0]),
        deadline_s=np.array([5.0]),
        lambda_Bps=np.array([3.0]),
        rho_prefill=np.array([4.0]),
        C_net=np.array([6.0]),
        C_prefill=np.array([8.0]),
        ell_net=np.array([1.5]),
        ell_prefill=np.array([4.0]),
        h_ctx=np.zeros((1, 1)),
        h_kv=np.zeros((1, 1)),
        retained_prefill_target_s=20.0,
    )

    scaled = _with_window(problem, 10.0)

    np.testing.assert_allclose(scaled.T, problem.T)
    assert not np.shares_memory(scaled.T, problem.T)
    assert scaled.retained_prefill_target_s == 20.0
    assert np.allclose(scaled.C_net, [30.0])
    assert np.allclose(scaled.C_prefill, [40.0])
    assert np.allclose(scaled.ell_net / scaled.C_net, problem.ell_net / problem.C_net)
    assert np.allclose(scaled.ell_prefill / scaled.C_prefill, problem.ell_prefill / problem.C_prefill)


def test_h1_stress_fraction_is_sampled_independent_of_frontier():
    assert STRESS_FRACTIONS == (0.25,)


def test_target_shortfall_keeps_solver_moved_amount_while_reporting_queue_diagnostics(monkeypatch):
    problem = SimpleNamespace(retained_prefill_target_s=10.0)
    solver = lambda _: SimpleNamespace(feasible=False, allocation=np.zeros((1, 2)), objective=None)
    captured = {}

    monkeypatch.setattr(retained, "with_retained_prefill_fraction", lambda base, fraction: problem)
    monkeypatch.setattr(
        retained,
        "_solver_metrics",
        lambda problem, y: {
            "retained_prefill_moved_s": 9.0,
            "resident_state_tb": 1.0,
            "average_equivalent_state_target_tb": 1.0,
            "actual_evacuated_state_tb": 0.9,
            "retained_prefill_moved_fraction": 0.09,
            "actual_evacuated_nvl72_hbm_fraction": 0.01,
            "request_migration_fraction": 0.2,
        },
    )
    monkeypatch.setattr(
        retained,
        "fractional_queue_load_proxy",
        lambda problem, y: {"fractional_network_capacity_pressure": 0.1, "fractional_prefill_capacity_pressure": 0.2},
    )
    def queue_metrics(*args, **kwargs):
        captured.update(kwargs)
        return _queue_metrics()

    monkeypatch.setattr(retained, "queue_metrics", queue_metrics)

    row = _policy_row(problem, "policy", solver, "edf", 7, 11, 0.25, 20.0)

    assert captured["drain_window_s"] == QUEUE_RELEASE_SPAN_S
    assert row["workload_seed"] == 11
    assert row["failure_mode"] == "target_not_met"
    assert row["retained_prefill_moved_s"] == 9.0
    assert row["retained_prefill_removal_rate_s_per_s"] == 11.0 / 20.0
    assert row["network_capacity_pressure"] == 1.2
    assert row["absolute_deadline_miss_rate"] == 0.4


def test_safe_definition_uses_absolute_event_start_deadline_metrics():
    metrics = {
        "retained_prefill_moved_s": 10.0,
        "retained_prefill_target_s": 10.0,
        "deadline_miss_rate": 0.5,
        "p95_reconstruction_delay_ratio": 3.0,
        "absolute_deadline_miss_rate": 0.0,
        "absolute_p95_delay_over_deadline": 1.0,
        "network_capacity_pressure": 1.0,
        "prefill_capacity_pressure": 1.0,
        "drain_completion_s": 10.0,
        "drain_window_s": 10.0,
    }

    assert _is_safe(metrics)
    assert not _is_safe({**metrics, "absolute_deadline_miss_rate": 0.011})
    assert not _is_safe({**metrics, "absolute_p95_delay_over_deadline": 1.001})
    assert not _is_safe({**metrics, "retained_prefill_moved_s": 9.99})
    assert not _is_safe({**metrics, "network_capacity_pressure": 1.0001})
    assert not _is_safe({**metrics, "prefill_capacity_pressure": 1.0001})
    assert not _is_safe({**metrics, "drain_completion_s": 10.0001})


def test_failure_mode_uses_explicit_frontier_failure_names():
    row = {
        "retained_prefill_moved_s": 9.0,
        "retained_prefill_target_s": 10.0,
        "deadline_miss_rate": 0.0,
        "p95_delay_over_deadline": 0.5,
        "absolute_deadline_miss_rate": 0.0,
        "absolute_p95_delay_over_deadline": 0.5,
        "network_capacity_pressure": 0.1,
        "prefill_capacity_pressure": 0.1,
    }

    assert _failure_mode(row) == "target_not_met"
    assert (
        _failure_mode({**row, "retained_prefill_moved_s": 10.0, "absolute_deadline_miss_rate": 0.02})
        == "absolute_deadline_miss"
    )
    assert (
        _failure_mode({**row, "retained_prefill_moved_s": 10.0, "absolute_p95_delay_over_deadline": 1.1})
        == "absolute_p95_delay"
    )
    assert _failure_mode({**row, "retained_prefill_moved_s": 10.0, "network_capacity_pressure": 1.1}) == "network_pressure"
    assert _failure_mode({**row, "retained_prefill_moved_s": 10.0, "prefill_capacity_pressure": 1.1}) == "prefill_pressure"
    assert (
        _failure_mode({**row, "retained_prefill_moved_s": 10.0, "drain_completion_s": 11.0, "drain_window_s": 10.0})
        == "drain_window_exceeded"
    )


def test_binary_search_validates_local_nonmonotone_safety(monkeypatch):
    def policy_row(base, policy, solver, release_policy, release_seed, workload_seed, fraction, drain_window_s):
        return _row(
            policy, release_policy, release_seed, workload_seed, drain_window_s, fraction, safe=fraction <= 0.47 or fraction == 0.49
        )

    monkeypatch.setattr(retained, "_policy_row", policy_row)

    rows, frontier = _frontier_job(("policy", object(), "edf", 7, 3, 10.0, object()))

    assert frontier["max_safe_retained_prefill_fraction"] == 0.49
    assert frontier["first_unsafe_retained_prefill_fraction"] == 0.5
    assert {row["retained_prefill_fraction"] for row in rows} >= {0.47, 0.48, 0.49, 0.5}


def test_frontier_rows_keep_each_absolute_deadline_window(monkeypatch):
    monkeypatch.setattr(retained, "FRONTIER_POLICIES", ("policy",))
    monkeypatch.setattr(retained, "FRONTIER_RELEASE_POLICIES", ("edf",))
    monkeypatch.setattr(retained, "DRAIN_WINDOWS_S", (10.0, 20.0, 40.0))
    rows = [
        _frontier("policy", "edf", "", 7, 10.0, 0.2),
        _frontier("policy", "edf", "", 7, 20.0, 0.5),
        _frontier("policy", "edf", "", 7, 40.0, 0.3),
    ]

    frontier = _monotone_frontier(rows)

    assert [row["max_safe_retained_prefill_fraction"] for row in frontier] == [0.2, 0.5, 0.3]
    assert [row["drain_window_s"] for row in frontier] == [10.0, 20.0, 40.0]


def test_frontier_rows_preserve_workload_seeds(monkeypatch):
    monkeypatch.setattr(retained, "FRONTIER_POLICIES", ("policy",))
    monkeypatch.setattr(retained, "FRONTIER_RELEASE_POLICIES", ("edf",))
    monkeypatch.setattr(retained, "DRAIN_WINDOWS_S", (10.0,))
    rows = [
        _frontier("policy", "edf", "", 2, 10.0, 0.2),
        _frontier("policy", "edf", "", 3, 10.0, 0.5),
    ]

    frontier = _monotone_frontier(rows)

    assert [(row["workload_seed"], row["max_safe_retained_prefill_fraction"]) for row in frontier] == [(2, 0.2), (3, 0.5)]


def test_run_pairs_problem_window_with_output_frontier_and_writes_drain_outputs(monkeypatch, tmp_path):
    built = []
    captured = {}

    def make_base(model, regime, **kwargs):
        built.append((kwargs["workload_seed"], kwargs["window_s"], kwargs["retained_prefill_fraction"]))
        return SimpleNamespace(workload_seed=kwargs["workload_seed"])

    def run_jobs(label, jobs, fn):
        assert label == "retained-state drain frontier"
        assert [(job[2], job[3], job[4], job[5], job[6].window_s) for job in jobs] == [
            ("edf", 7, 2, 10.0, 10.0),
            ("edf", 7, 2, 20.0, 20.0),
            ("edf", 7, 3, 10.0, 10.0),
            ("edf", 7, 3, 20.0, 20.0),
        ]
        return [
            (
                [_row(policy, release_policy, release_seed, workload_seed, drain_window_s, 0.5, safe=True)],
                _frontier(policy, release_policy, "", workload_seed, drain_window_s, 0.5),
            )
            for policy, _, release_policy, release_seed, workload_seed, drain_window_s, _ in jobs
        ]

    monkeypatch.setattr(retained, "ROOT", tmp_path)
    monkeypatch.setattr(retained, "DRAIN_WINDOWS_S", (10.0, 20.0))
    monkeypatch.setattr(retained, "FRONTIER_RELEASE_POLICIES", ("edf",))
    monkeypatch.setattr(retained, "POLICIES", (("policy", lambda problem: None),))
    monkeypatch.setattr(retained, "FRONTIER_POLICIES", ("policy",))
    monkeypatch.setattr(retained, "make_problem", make_base)
    monkeypatch.setattr(retained, "_with_window", lambda base, window_s: SimpleNamespace(window_s=window_s))
    monkeypatch.setattr(retained, "_run_jobs", run_jobs)
    monkeypatch.setattr(retained, "_write_rows", lambda path, rows, columns: captured.setdefault(path.name, rows))
    monkeypatch.setattr(retained, "_print_latex_frontier", lambda rows: None)
    monkeypatch.setattr(retained, "_print_diagnostics", lambda rows: None)

    rows, frontier = retained.run_retained_state_frontier(WorkloadConfig(source="fixed"), workload_seeds=(2, 3))

    assert built == [(2, 1.0, 1.0), (3, 1.0, 1.0)]
    assert [(row["release_policy"], row["workload_seed"], row["drain_window_s"]) for row in rows] == [
        ("edf", 2, 10.0),
        ("edf", 2, 20.0),
        ("edf", 3, 10.0),
        ("edf", 3, 20.0),
    ]
    assert [(row["release_policy"], row["workload_seed"], row["drain_window_s"]) for row in frontier] == [
        ("edf", 2, 10.0),
        ("edf", 2, 20.0),
        ("edf", 3, 10.0),
        ("edf", 3, 20.0),
    ]
    assert set(captured) == {"retained_state_drain_sweep.csv", "retained_state_drain_frontier.csv"}


def _row(policy, release_policy, release_seed, workload_seed, drain_window_s, retained_prefill_fraction, safe):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": release_seed if release_policy == "random" else "",
        "workload_seed": workload_seed,
        "drain_window_s": drain_window_s,
        "deadline_scale": 1.0,
        "retained_prefill_fraction": retained_prefill_fraction,
        "status": "SAFE" if safe else "UNSAFE",
        "safe": safe,
        "failure_mode": "" if safe else "absolute_p95_delay",
        "retained_prefill_target_s": 10.0,
        "retained_prefill_moved_s": 10.0,
        "resident_state_tb": 1.0,
        "average_equivalent_state_target_tb": 2.0,
        "actual_evacuated_state_tb": 3.0,
        "retained_prefill_moved_fraction": retained_prefill_fraction,
        "actual_evacuated_nvl72_hbm_fraction": 3.0 / 13.4,
        "retained_prefill_removal_rate_s_per_s": 10.0 / drain_window_s,
        "request_migration_fraction": 0.2,
        "mean_delay_s": 1.0,
        "p50_delay_s": 1.0,
        "p95_delay_s": 2.0,
        "p99_delay_s": 2.0,
        "p95_delay_over_deadline": 0.2,
        "deadline_miss_rate": 0.0,
        "absolute_p95_delay_over_deadline": 0.9 if safe else 1.1,
        "absolute_deadline_miss_rate": 0.0,
        "network_capacity_pressure": 0.2,
        "prefill_capacity_pressure": 0.3,
        "replay_retained_prefill_fraction": 0.25,
        "state_transfer_retained_prefill_fraction": 0.75,
        "fractional_network_capacity_pressure": 0.2,
        "fractional_prefill_capacity_pressure": 0.3,
        "drain_completion_s": drain_window_s,
        "objective": 1.0,
    }


def _frontier(policy, release_policy, release_seed, workload_seed, drain_window_s, fraction):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": release_seed,
        "workload_seed": workload_seed,
        "drain_window_s": drain_window_s,
        "deadline_scale": 1.0,
        "max_safe_retained_prefill_fraction": fraction,
        "frontier_censored_by_search": False,
        "first_unsafe_retained_prefill_fraction": fraction + 0.01,
        "first_unsafe_failure_mode": "absolute_p95_delay",
        "absolute_p95_delay_over_deadline_at_frontier": 0.9,
        "absolute_deadline_miss_rate_at_frontier": 0.0,
        "p95_delay_over_deadline_at_frontier": 0.2,
        "deadline_miss_rate_at_frontier": 0.0,
        "network_capacity_pressure_at_frontier": 0.2,
        "prefill_capacity_pressure_at_frontier": 0.3,
        "average_equivalent_state_target_tb_at_frontier": 2.0,
        "actual_evacuated_state_tb_at_frontier": 3.0,
        "actual_evacuated_nvl72_hbm_fraction_at_frontier": 3.0 / 13.4,
        "replay_retained_prefill_fraction_at_frontier": 0.25,
        "state_transfer_retained_prefill_fraction_at_frontier": 0.75,
        "drain_completion_s_at_frontier": drain_window_s,
    }


def _queue_metrics():
    return {
        "retained_prefill_moved_s": 11.0,
        "retained_prefill_removal_rate_s_per_s": 0.5,
        "mean_reconstruction_delay": 3.0,
        "p50_reconstruction_delay": 3.0,
        "p95_reconstruction_delay": 4.0,
        "p99_reconstruction_delay": 4.0,
        "p95_reconstruction_delay_ratio": 0.8,
        "deadline_miss_rate": 0.0,
        "absolute_p95_delay_over_deadline": 1.4,
        "absolute_deadline_miss_rate": 0.4,
        "network_capacity_pressure": 1.2,
        "prefill_capacity_pressure": 0.3,
        "resident_state_tb": 1.0,
        "average_equivalent_state_target_tb": 1.0,
        "actual_evacuated_state_tb": 1.1,
        "retained_prefill_moved_fraction": 0.11,
        "actual_evacuated_nvl72_hbm_fraction": 0.02,
        "replay_retained_prefill_fraction": 1.0,
        "state_transfer_retained_prefill_fraction": 0.0,
        "drain_completion_s": 19.0,
    }
