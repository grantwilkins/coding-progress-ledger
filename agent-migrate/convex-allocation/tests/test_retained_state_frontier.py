"""
Claim:
The retained-state drain frontier reports the largest rounded queue-safe
retained-prefill fraction for each allocation policy, release policy, and drain
window, using absolute event-start deadlines while tying solver capacity and
queue drain horizons to the same x-value.

Plausible wrong implementations:
- Keep using release-relative reconstruction safety after adding drain-order
  ablations.
- Vary drain_window_s in queue simulation but leave make_problem(window_s) fixed.
- Trust binary search monotonicity despite rounded nonmonotone safety.
- Drop the least-loaded ordinary-routing baseline from the north-star plot.
- Collapse release-policy rows and hide EDF/order-oblivious differences.
- Keep writing the old deadline-scale frontier CSV names.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import experiments.run_retained_state_frontier as retained
from experiments.run_retained_state_frontier import (
    DRAIN_WINDOWS_S,
    FRONTIER_POLICIES,
    MAIN_POLICY,
    _failure_mode,
    _frontier_job,
    _is_safe,
    _monotone_frontier,
)
from evaluation import WorkloadConfig


def test_drain_windows_are_positive_log_plot_points():
    assert DRAIN_WINDOWS_S == (10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1200.0, 2400.0, 3600.0)
    assert all(window > 0.0 for window in DRAIN_WINDOWS_S)


def test_frontier_policy_set_includes_least_loaded_baseline():
    assert MAIN_POLICY == "deadline-penalty-rounded"
    assert FRONTIER_POLICIES == (
        "deadline-penalty-rounded",
        "online-queue-greedy",
        "least-loaded-destination",
        "replay-only",
        "state-only",
    )


def test_safe_definition_uses_absolute_event_start_deadline_metrics():
    metrics = {
        "retained_prefill_moved_s": 10.0,
        "retained_prefill_target_s": 10.0,
        "deadline_miss_rate": 0.5,
        "p95_reconstruction_delay_ratio": 3.0,
        "absolute_deadline_miss_rate": 0.0,
        "absolute_p95_delay_over_deadline": 1.0,
    }

    assert _is_safe(metrics)
    assert not _is_safe({**metrics, "absolute_deadline_miss_rate": 0.011})
    assert not _is_safe({**metrics, "absolute_p95_delay_over_deadline": 1.001})
    assert not _is_safe({**metrics, "retained_prefill_moved_s": 9.99})


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


def test_binary_search_validates_local_nonmonotone_safety(monkeypatch):
    def policy_row(base, policy, solver, release_policy, release_seed, fraction, drain_window_s):
        return _row(policy, release_policy, release_seed, drain_window_s, fraction, safe=fraction <= 0.47 or fraction == 0.49)

    monkeypatch.setattr(retained, "_policy_row", policy_row)

    rows, frontier = _frontier_job(("policy", object(), "edf", 7, 10.0, object()))

    assert frontier["max_safe_retained_prefill_fraction"] == 0.49
    assert frontier["first_unsafe_retained_prefill_fraction"] == 0.5
    assert {row["retained_prefill_fraction"] for row in rows} >= {0.47, 0.48, 0.49, 0.5}


def test_frontier_rows_are_monotone_available_window_envelope(monkeypatch):
    monkeypatch.setattr(retained, "FRONTIER_POLICIES", ("policy",))
    monkeypatch.setattr(retained, "RELEASE_POLICIES", ("edf",))
    monkeypatch.setattr(retained, "DRAIN_WINDOWS_S", (10.0, 20.0, 40.0))
    rows = [
        _frontier("policy", "edf", "", 10.0, 0.2),
        _frontier("policy", "edf", "", 20.0, 0.5),
        _frontier("policy", "edf", "", 40.0, 0.3),
    ]

    frontier = _monotone_frontier(rows)

    assert [row["max_safe_retained_prefill_fraction"] for row in frontier] == [0.2, 0.5, 0.5]
    assert [row["drain_window_s"] for row in frontier] == [10.0, 20.0, 40.0]


def test_run_pairs_problem_window_with_queue_drain_and_writes_drain_outputs(monkeypatch, tmp_path):
    built = []
    captured = {}

    def make_base(model, regime, **kwargs):
        built.append((kwargs["window_s"], kwargs["retained_prefill_fraction"]))
        return SimpleNamespace(window_s=kwargs["window_s"])

    def run_jobs(label, jobs, fn):
        assert label == "retained-state drain frontier"
        assert [(job[2], job[4], job[5].window_s) for job in jobs] == [
            ("edf", 10.0, 10.0),
            ("random", 10.0, 10.0),
            ("edf", 20.0, 20.0),
            ("random", 20.0, 20.0),
        ]
        return [
            (
                [_row(policy, release_policy, release_seed, drain_window_s, 0.5, safe=True)],
                _frontier(policy, release_policy, 7 if release_policy == "random" else "", drain_window_s, 0.5),
            )
            for policy, _, release_policy, release_seed, drain_window_s, _ in jobs
        ]

    monkeypatch.setattr(retained, "ROOT", tmp_path)
    monkeypatch.setattr(retained, "DRAIN_WINDOWS_S", (10.0, 20.0))
    monkeypatch.setattr(retained, "RELEASE_POLICIES", ("edf", "random"))
    monkeypatch.setattr(retained, "POLICIES", (("policy", lambda problem: None),))
    monkeypatch.setattr(retained, "FRONTIER_POLICIES", ("policy",))
    monkeypatch.setattr(retained, "make_problem", make_base)
    monkeypatch.setattr(retained, "_run_jobs", run_jobs)
    monkeypatch.setattr(retained, "_write_rows", lambda path, rows, columns: captured.setdefault(path.name, rows))
    monkeypatch.setattr(retained, "_print_latex_frontier", lambda rows: None)
    monkeypatch.setattr(retained, "_print_diagnostics", lambda rows: None)

    rows, frontier = retained.run_retained_state_frontier(WorkloadConfig(source="fixed"))

    assert built == [(10.0, 1.0), (10.0, 1.0), (20.0, 1.0), (20.0, 1.0)]
    assert [(row["release_policy"], row["drain_window_s"]) for row in rows] == [
        ("edf", 10.0),
        ("random", 10.0),
        ("edf", 20.0),
        ("random", 20.0),
    ]
    assert [(row["release_policy"], row["drain_window_s"]) for row in frontier] == [
        ("edf", 10.0),
        ("edf", 20.0),
        ("random", 10.0),
        ("random", 20.0),
    ]
    assert set(captured) == {"retained_state_drain_sweep.csv", "retained_state_drain_frontier.csv"}


def _row(policy, release_policy, release_seed, drain_window_s, retained_prefill_fraction, safe):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": release_seed if release_policy == "random" else "",
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


def _frontier(policy, release_policy, release_seed, drain_window_s, fraction):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "release_seed": release_seed,
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
