"""
Claim:
The report figure script emits exactly one simple artifact per hypothesis plus a
compact integer benchmark table. The H1 table uses one fixed stress target and
absolute event-start deadline metrics. The H2 frontier compares allocation
policies under EDF release with workload-seed error bars and independent
drain-window points, the H2 CDF is request-level in reconstruction-delay
seconds, and the H4 heatmap exposes per-class state locality from the same
workload config rather than only destination load. The H3 plot is the direct
single-request replay/state crossover implied by model architecture.

Plausible wrong implementations:
- Reintroduce crowded diagnostic plots instead of the five report figures.
- Build H1 from frontier rows or mixed release policies instead of one target.
- Let a target shortfall pass because pressure and delay are low.
- Keep using release-relative reconstruction metrics after switching drain
  frontier safety to event-start deadlines.
- Restore an available-window frontier envelope that hides independent
  drain-window outcomes.
- Drop allocation-policy lines from H2.
- Use release-order seeds instead of varied workload seeds for H2 error bars.
- Let H2 mark overloaded or over-window rows safe after frontier safety changes.
- Average class delays before building the CDF.
- Plot delay/deadline ratios after the CDF is supposed to use seconds.
- Flip the H3 crossover inequality, drop the bytes-to-bits conversion, or let
  context length/request count change the replay-vs-state decision.
- Keep unrelated integer policies in the compact summary table.
- Drop context/KV locality from the manifest heatmap labels.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from catalog import ModelParams
from experiments.plot_queue_centered import (
    FRONTIER_RELEASE_POLICY,
    H3_CONTEXT_TOKENS,
    H3_REQUEST_COUNT,
    INTEGER_TABLE_POLICIES,
    OUTPUT_FILES,
    REPORT_POLICIES,
    _allocation_heatmap,
    _architecture_action_mix,
    _clear_outputs,
    _cdf_points,
    _delay_seconds,
    _frontier_capped_retained_fraction,
    _h3_action_rows,
    _h1_stress_rows,
    _integer_summary_rows,
    _max_waiting_depth_points,
    _plot_state_manifest_heatmap,
    _safe_frontier,
    _safe_series,
)
from evaluation import WorkloadConfig
from problem import ProblemData

import numpy as np


def test_report_outputs_are_one_plot_per_hypothesis_plus_summary_table():
    assert set(OUTPUT_FILES) == {
        "h1_fixed_target_stress.csv",
        "h2_safe_frontier.csv",
        "h2_safe_frontier.pdf",
        "h2_delay_cdf.pdf",
        "h3_action_mix_by_model.csv",
        "h3_action_mix_by_model.pdf",
        "h4_state_manifest_heatmap.pdf",
        "integer_benchmark_summary.csv",
    }
    assert not any("queue_depth" in path or "busy_scatter" in path for path in OUTPUT_FILES)


def test_report_policies_are_small_enough_to_read_without_overplotting():
    assert REPORT_POLICIES == (
        "deadline-penalty-rounded",
        "online-queue-greedy",
        "least-loaded-destination",
        "replay-only",
        "state-only",
    )
    assert len(REPORT_POLICIES) == 5


def test_cdf_example_fraction_is_capped_by_safe_frontier():
    frontier = pd.DataFrame(
        {
            "policy": ["deadline-penalty-rounded", "deadline-penalty-rounded"],
            "drain_window_s": [1000.0, 1200.0],
            "max_safe_retained_prefill_fraction": [1.0, 0.25],
        }
    )

    assert _frontier_capped_retained_fraction(frontier, "deadline-penalty-rounded", 0.5) == 0.25


def test_cdf_example_fraction_keeps_requested_fraction_without_frontier_point():
    frontier = pd.DataFrame(
        {
            "policy": ["deadline-penalty-rounded"],
            "drain_window_s": [1000.0],
            "max_safe_retained_prefill_fraction": [0.25],
        }
    )

    assert _frontier_capped_retained_fraction(frontier, "deadline-penalty-rounded", 0.5) == 0.5


def test_safe_series_requires_absolute_deadline_bounds():
    df = pd.DataFrame(
        {
            "absolute_deadline_miss_rate": [0.01, 0.011, 0.0],
                "absolute_p95_delay_over_deadline": [1.0, 0.5, 1.001],
                "retained_prefill_moved_s": [10.0, 10.0, 10.0],
                "retained_prefill_target_s": [10.0, 10.0, 10.0],
                "network_capacity_pressure": [1.0, 1.0, 1.0],
                "prefill_capacity_pressure": [1.0, 1.0, 1.0],
                "drain_completion_s": [10.0, 10.0, 10.0],
                "drain_window_s": [10.0, 10.0, 10.0],
            }
        )

    assert _safe_series(df).tolist() == [True, False, False]


def test_safe_series_ignores_release_relative_deadline_metrics_when_present():
    df = pd.DataFrame(
        {
            "deadline_miss_rate": [0.5],
            "p95_delay_over_deadline": [3.0],
            "absolute_deadline_miss_rate": [0.0],
                "absolute_p95_delay_over_deadline": [0.1],
                "retained_prefill_moved_s": [10.0],
                "retained_prefill_target_s": [10.0],
                "network_capacity_pressure": [1.0],
                "prefill_capacity_pressure": [1.0],
                "drain_completion_s": [10.0],
                "drain_window_s": [10.0],
            }
        )

    assert _safe_series(df).tolist() == [True]


def test_safe_series_rejects_target_shortfall_even_when_deadlines_pass():
    df = pd.DataFrame(
        {
            "absolute_deadline_miss_rate": [0.0],
                "absolute_p95_delay_over_deadline": [0.5],
                "retained_prefill_moved_s": [9.0],
                "retained_prefill_target_s": [10.0],
                "network_capacity_pressure": [1.0],
                "prefill_capacity_pressure": [1.0],
                "drain_completion_s": [10.0],
                "drain_window_s": [10.0],
            }
        )

    assert _safe_series(df).tolist() == [False]


def test_safe_frontier_uses_largest_safe_fraction_by_drain_window():
    rows = [
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.2, 0.0, 0.8),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.4, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.6, 0.2, 0.9),
        {**_sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.8, 0.0, 0.9), "network_capacity_pressure": "1.1"},
        {**_sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.9, 0.0, 0.9), "drain_completion_s": "901.0"},
        _sweep_row("deadline-penalty-rounded", "edf", 1800.0, 0.6, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "edf", 3600.0, 0.3, 0.0, 0.9),
        _sweep_row("deadline-penalty-rounded", "random", 900.0, 0.3, 0.0, 0.9),
        _sweep_row("replay-only", "edf", 900.0, 0.9, 0.0, 0.9),
    ]

    frontier = _safe_frontier(rows)
    by_policy_window = {
        (row.policy, row.drain_window_s): row.max_safe_retained_prefill_fraction
        for row in frontier.itertuples()
    }

    assert FRONTIER_RELEASE_POLICY == "edf"
    assert by_policy_window[("deadline-penalty-rounded", 900.0)] == 0.4
    assert by_policy_window[("deadline-penalty-rounded", 1800.0)] == 0.6
    assert by_policy_window[("deadline-penalty-rounded", 3600.0)] == 0.3
    assert by_policy_window[("replay-only", 900.0)] == 0.9
    assert set(frontier["policy"]) == {"deadline-penalty-rounded", "replay-only"}
    assert set(frontier["max_safe_retained_prefill_fraction_std"]) == {0.0}
    assert frontier[frontier["policy"] == "deadline-penalty-rounded"][
        "max_safe_retained_prefill_fraction"
    ].tolist() == [0.4, 0.6, 0.3]


def test_safe_frontier_averages_workload_seed_frontiers_and_reports_std():
    rows = [
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.2, 0.0, 0.8, workload_seed=1),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.4, 0.0, 0.9, workload_seed=1),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.6, 0.2, 0.9, workload_seed=1),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.2, 0.0, 0.8, workload_seed=2),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.4, 0.2, 0.9, workload_seed=2),
        _sweep_row("deadline-penalty-rounded", "edf", 900.0, 0.2, 0.0, 0.8),
    ]

    row = _safe_frontier(rows).query("policy == 'deadline-penalty-rounded'").iloc[0]

    assert np.isclose(row.max_safe_retained_prefill_fraction, (0.4 + 0.2 + 0.2) / 3)
    assert row.seed_count == 3
    assert row.max_safe_retained_prefill_fraction_std > 0.0


def test_h1_fixed_target_stress_table_uses_one_edf_target_row_per_policy():
    rows = [
        _h1_row("deadline-penalty-rounded", "edf", 20.0, 0.25, 10.0, 10.0, 0.8, 0.7, 0.9, 0.0),
        _h1_row("online-queue-greedy", "edf", 20.0, 0.25, 8.0, 10.0, 0.2, 0.2, 0.4, 0.0),
        _h1_row("least-loaded-destination", "edf", 20.0, 0.25, 7.0, 10.0, 0.1, 0.1, 0.2, 0.0),
        _h1_row("replay-only", "edf", 20.0, 0.25, 9.0, 10.0, 0.0, 0.9, 0.8, 0.0),
        _h1_row("state-only", "edf", 20.0, 0.25, 11.0, 10.0, 1.2, 0.0, 0.5, 0.0),
        _h1_row("deadline-penalty-rounded", "random", 20.0, 0.25, 1.0, 10.0, 0.0, 0.0, 0.1, 0.0),
        _h1_row("deadline-penalty-rounded", "edf", 40.0, 0.25, 1.0, 10.0, 0.0, 0.0, 0.1, 0.0),
        _h1_row("deadline-penalty-rounded", "edf", 20.0, 0.5, 1.0, 10.0, 0.0, 0.0, 0.1, 0.0),
    ]

    table = _h1_stress_rows(rows, deadline_scale=1.0)
    by_policy = {row["policy"]: row for row in table}

    assert [row["release_policy"] for row in table] == ["edf"] * 5
    assert [row["drain_window_s"] for row in table] == [20.0] * 5
    assert by_policy["Deadline-aware"]["verdict"] == "Pass"
    assert by_policy["Online queue"]["verdict"] == "Target shortfall"
    assert by_policy["Least loaded"]["verdict"] == "Target shortfall"
    assert by_policy["Replay only"]["verdict"] == "Target shortfall"
    assert by_policy["State only"]["verdict"] == "Network overload"


def test_cdf_points_are_request_level_empirical_cdf():
    assert _cdf_points([3.0, 1.0, 3.0]) == [(1.0, 1 / 3), (3.0, 2 / 3), (3.0, 1.0)]


def test_delay_cdf_uses_seconds_not_deadline_normalized_ratios():
    trace = [
        SimpleNamespace(reconstruction_delay=8.0, deadline_s=4.0),
        SimpleNamespace(reconstruction_delay=3.0, deadline_s=30.0),
    ]

    assert _cdf_points(_delay_seconds(trace)) == [(3.0, 0.5), (8.0, 1.0)]


def test_h3_crossover_uses_model_bytes_prefill_and_network_units():
    model = ModelParams("toy", beta_bytes_per_tok=4.0, eta_bytes_per_tok=1004.0, prefill_tok_s=125_000.0, reference_crossover_gbps=1.0)
    rows = _h3_action_rows((model,), context_tokens=2_000, request_count=3, network_gbps=(0.5, 1.0, 2.0))
    by_gbps = {row["network_throughput_gbps"]: row for row in rows}

    assert by_gbps[0.5]["crossover_gbps"] == 1.0
    assert by_gbps[0.5]["replay_time_s"] < by_gbps[0.5]["state_transfer_time_s"]
    assert by_gbps[0.5]["replay_fraction"] == 1.0
    assert by_gbps[1.0]["replay_fraction"] == 1.0
    assert by_gbps[2.0]["state_transfer_time_s"] < by_gbps[2.0]["replay_time_s"]
    assert by_gbps[2.0]["replay_fraction"] == 0.0


def test_h3_context_and_request_count_scale_times_not_decisions():
    model = ModelParams("toy", beta_bytes_per_tok=4.0, eta_bytes_per_tok=1004.0, prefill_tok_s=125_000.0, reference_crossover_gbps=1.0)
    small = _h3_action_rows((model,), context_tokens=1_000, request_count=2, network_gbps=(0.5,))[0]
    large = _h3_action_rows((model,), context_tokens=2_000, request_count=3, network_gbps=(0.5,))[0]

    assert large["replay_time_s"] / small["replay_time_s"] == 3.0
    assert large["state_transfer_time_s"] / small["state_transfer_time_s"] == 3.0
    assert large["replay_fraction"] == small["replay_fraction"] == 1.0


def test_h3_catalog_architectures_produce_different_actions_at_same_network():
    df = _architecture_action_mix()
    actions_at_3g = df[df["network_throughput_gbps"] == 3.0].set_index("model")["replay_fraction"].to_dict()
    actions_at_20g = df[df["network_throughput_gbps"] == 20.0].set_index("model")["replay_fraction"].to_dict()

    assert H3_CONTEXT_TOKENS == 128_000
    assert H3_REQUEST_COUNT == 1_000
    assert actions_at_3g == {"DeepSeek-V4-Pro": 0.0, "GLM-5": 1.0, "Qwen3-Next-80B-A3B": 1.0}
    assert actions_at_20g == {"DeepSeek-V4-Pro": 0.0, "GLM-5": 0.0, "Qwen3-Next-80B-A3B": 1.0}


def test_integer_summary_keeps_only_compact_methodology_rows():
    rows = [
        _integer_row("case-a", policy)
        for policy in (*INTEGER_TABLE_POLICIES, "online-queue-greedy", "state-only")
    ]

    summary = _integer_summary_rows(rows)

    assert [row["policy"] for row in summary] == list(INTEGER_TABLE_POLICIES)
    assert set(summary[0]) == {"case", "policy", "integer_objective_gap_to_best", "p95_delay", "miss_rate"}


def test_manifest_heatmap_labels_include_context_deadline_and_locality():
    model = ModelParams("manifest", 1.0, 2.0, 1.0, 0.0)
    problem = ProblemData(
        model=model,
        regime="manifest",
        T=np.array([1000.0, 2000.0]),
        d=np.array([2.0, 2.0]),
        deadline_s=np.array([10.0, 5.0]),
        lambda_Bps=np.array([10.0]),
        rho_prefill=np.array([10.0]),
        C_net=np.array([100.0]),
        C_prefill=np.array([100.0]),
        ell_net=np.zeros(1),
        ell_prefill=np.zeros(1),
        h_ctx=np.array([[0.2], [0.8]]),
        h_kv=np.array([[0.1], [0.5]]),
        retained_prefill_target_s=1.0,
    )
    allocation = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    heatmap, row_labels, col_labels = _allocation_heatmap(problem, allocation)

    assert heatmap.shape == (2, 3)
    assert col_labels == ["site 0\nreplay", "site 0\nstate", "stay"]
    assert all("T=" in label and "ddl=" in label and "ctx=" in label and "kv=" in label for label in row_labels)


def test_h4_heatmap_uses_the_report_workload_config(monkeypatch, tmp_path):
    captured = {}

    def make_problem(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("experiments.plot_queue_centered.make_problem", make_problem)
    monkeypatch.setattr("experiments.plot_queue_centered.solve_soft_deadline_cvxpy", lambda problem: SimpleNamespace(y=np.zeros((1, 1))))
    monkeypatch.setattr(
        "experiments.plot_queue_centered._allocation_heatmap",
        lambda problem, allocation, max_rows=6: (np.ones((1, 1)), ["class"], ["stay"]),
    )

    _plot_state_manifest_heatmap(
        WorkloadConfig(source="generated", seed=13, jobs=25, classes=5),
        tmp_path / "h4.pdf",
        retained_prefill_fraction=0.5,
        deadline_scale=1.0,
        frontier=pd.DataFrame(
            {
                "policy": ["deadline-penalty-rounded"],
                "drain_window_s": [1200.0],
                "max_safe_retained_prefill_fraction": [0.5],
            }
        ),
    )

    assert captured["workload_source"] == "generated"
    assert captured["workload_seed"] == 13
    assert captured["workload_jobs"] == 25
    assert captured["workload_classes"] == 5
    assert captured["retained_prefill_fraction"] == 0.5


def test_plot_output_cleanup_removes_owned_artifacts_only(tmp_path):
    owned = tmp_path / "h2_safe_frontier.csv"
    unrelated = tmp_path / "notes.csv"
    owned.write_text("stale")
    unrelated.write_text("keep")

    _clear_outputs(tmp_path)

    assert not owned.exists()
    assert unrelated.read_text() == "keep"


def test_queue_depth_helper_still_counts_waiting_requests_for_bandwidth_plot():
    trace = (
        _record(k=0, network_wait=0.0, network_service=1.0),
        _record(k=0, network_wait=2.0, network_service=1.0),
        _record(k=0, network_wait=4.0, network_service=1.0),
        _record(k=1, network_wait=3.0, network_service=1.0),
    )

    assert _max_waiting_depth_points(trace, "network") == [
        (0.0, 0.0),
        (0.0, 2.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (4.0, 0.0),
    ]


def _sweep_row(policy, release_policy, drain_window_s, retained_prefill_fraction, miss_rate, delay_ratio, workload_seed=""):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "workload_seed": str(workload_seed),
        "retained_prefill_fraction": str(retained_prefill_fraction),
        "deadline_scale": "0.5",
        "drain_window_s": str(drain_window_s),
        "network_capacity_pressure": "0.5",
        "prefill_capacity_pressure": "0.5",
        "deadline_miss_rate": str(miss_rate),
        "p95_delay_over_deadline": str(delay_ratio),
        "absolute_deadline_miss_rate": str(miss_rate),
        "absolute_p95_delay_over_deadline": str(delay_ratio),
        "retained_prefill_moved_s": "10.0",
        "retained_prefill_target_s": "10.0",
        "drain_completion_s": str(drain_window_s),
    }


def _h1_row(
    policy,
    release_policy,
    drain_window_s,
    retained_prefill_fraction,
    moved,
    target,
    net,
    prefill,
    p95,
    miss,
):
    return {
        "policy": policy,
        "release_policy": release_policy,
        "drain_window_s": str(drain_window_s),
        "deadline_scale": "1.0",
        "retained_prefill_fraction": str(retained_prefill_fraction),
        "retained_prefill_moved_s": str(moved),
        "retained_prefill_target_s": str(target),
        "network_capacity_pressure": str(net),
        "prefill_capacity_pressure": str(prefill),
        "absolute_p95_delay_over_deadline": str(p95),
        "absolute_deadline_miss_rate": str(miss),
    }


def _integer_row(case, policy):
    return {
        "case": case,
        "policy": policy,
        "integer_objective_gap_to_best": "0.1",
        "p95_delay": "2.0",
        "miss_rate": "0.0",
        "integer_objective": "1.0",
    }


def _record(k, network_wait, network_service, release_time=0.0):
    return SimpleNamespace(
        k=k,
        release_time_s=release_time,
        network_queue_wait=network_wait,
        network_service_time=network_service,
        prefill_queue_wait=0.0,
        prefill_service_time=0.0,
    )
