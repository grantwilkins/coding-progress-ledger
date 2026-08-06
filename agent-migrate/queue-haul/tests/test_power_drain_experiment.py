"""
Claim:
The experiment samples complete sessions, uses absolute local power limits, plans
once centrally, and preserves raw execution evidence for every uncertainty case.

Plausible wrong implementations:
- Resample independent workload fields and create impossible sessions.
- Re-plan the faster/slower cases and hide plan sensitivity.
- Reorder or resample scenarios when worker processes execute them.
- Test the last power sample instead of integrating the deadline window.
- Emit only a summary and make timing or network claims impossible to audit.
- Drop active/cold GPU residency while constructing simulator sessions.
- Size instances from compute while silently exceeding measured resident KV capacity.
- Omit destination KV queue evidence needed to explain migration time.
- Drop planned pool debt, recovery, or binding-resource evidence from summaries.
- Drop the physical resource ledger or selected destination pool from raw output.
- Fail to write plots when a valid plan contains no migrations.
- Let an idle migration snapshot generate requests or context growth.
"""

import csv
from pathlib import Path
import math
from dataclasses import replace

import pytest

import power_drain_experiment as experiment
from planner import source_power
from profiles import ModelProfile, WorkloadProfile
from simulate import QueueExecution


def test_build_scenario_packs_calibrated_sessions_and_named_links():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    scenario, route = experiment.build_scenario(workload, model, 12, 3, 500, 5, 5)

    assert len(scenario.sessions) == 12
    assert source_power(scenario, model) > 500
    assert all(len(route(f"source-{i}", "dest-0")) == 5
               for i in range(len(scenario.instances) // 2))
    assert all("source-dc-egress" in route(f"source-{i}", "dest-0")
               for i in range(len(scenario.instances) // 2))
    assert {node.site_id for node in scenario.nodes} == {"source-dc", "destination-dc"}


def test_active_load_does_not_use_cold_reactivation_probability():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    records = workload.sample(6, 3)
    scenario, _ = experiment.build_scenario(
        workload, model, 6, 3, 500, 5, 5, controller_delay_s=1
    )
    for session, record in zip(scenario.sessions, records):
        case = model.case()
        cycle = (record.request_gap_s + record.tool_delay_s
                 + record.prompt_tokens / case.prefill.rate(record.context_tokens, 1)
                 + record.output_tokens
                 / case.decode.rate(record.context_tokens, 1))
        assert session.expected_f == pytest.approx(record.prompt_tokens / cycle)
        assert session.expected_growth_tokens_per_s == pytest.approx(
            (record.prompt_tokens + record.output_tokens) / cycle
            * (1 + workload.source.relative_error)
        )
        assert session.wake_probability == 0


def test_idle_snapshot_preserves_sample_and_disables_future_activity():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[0])
    scenario, _ = experiment.build_scenario(
        workload, model, 12, 3, 0, 60, 60, idle_snapshot=True,
    )
    records = workload.sample(12, 3)

    assert [session.context_tokens for session in scenario.sessions] == [
        record.context_tokens for record in records
    ]
    assert [session.log_bytes for session in scenario.sessions] == [
        record.log_bytes for record in records
    ]
    assert all(not session.requests and not session.expected_growth_tokens_per_s
               for session in scenario.sessions)


def test_scenario_preserves_cold_sessions_without_gpu_load():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    workload = replace(workload, records=(replace(workload.records[0], state="cold"),))
    scenario, _ = experiment.build_scenario(workload, model, 1, 3, 500, 5, 5)
    session = scenario.sessions[0]

    assert session.state == "cold"
    assert session.expected_f == session.expected_g == 0
    assert len(session.requests) <= 1
    assert session.wake_probability == pytest.approx(
        1 - math.exp(-(5 - workload.records[0].tool_delay_s)
                     / workload.records[0].request_gap_s)
    )


def test_scenario_packing_enforces_compute_and_resident_kv_capacity():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[0])
    record = workload.records[0]
    profile = replace(model, kv_capacity_tokens=2 * record.context_tokens)
    active, _ = experiment.build_scenario(
        replace(workload, records=(record,)), profile, 3, 0, 0, 5, 5
    )
    cold, _ = experiment.build_scenario(
        replace(workload, records=(replace(record, state="cold"),)),
        profile, 3, 0, 0, 5, 5,
    )

    assert len(active.instances) // 2 == 2
    assert len(cold.instances) // 2 == 1


def test_profile_range_is_checked_before_a_long_run():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    with pytest.raises(ValueError, match="measured context range"):
        experiment.build_scenario(workload, model, 6, 0, 500, 120, 180)


def test_scenario_builder_hard_fails_unmodeled_tensor_parallel_topology():
    model = replace(ModelProfile.load(experiment.DEFAULT_MODEL), tensor_parallel=2)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    with pytest.raises(ValueError, match="tensor parallel size 1"):
        experiment.build_scenario(workload, model, 6, 3, 500, 5, 5)


def test_excess_energy_integrates_step_power_after_deadline():
    power = ((0, 100, 0), (6, 40, 0), (8, 20, 0))
    assert experiment.excess_energy(power, 5, 10, 30) == pytest.approx(90)


def test_queue_summary_counts_pending_wait_through_the_cutoff():
    rows = (
        QueueExecution("active", "initial", "dest", 100, 0, 0, 1, 0, 0),
        QueueExecution("pending", "initial", "dest", 100, 0, None, None, 1, 100),
    )
    assert experiment.queue_summary(rows, 1) == {
        "destination_kv_queue_operations": 2,
        "destination_kv_queue_max_depth": 1,
        "destination_kv_queue_max_bytes": 100,
        "destination_kv_queue_pending_at_end": 1,
        "destination_kv_queue_pending_bytes_at_end": 100,
        "destination_kv_queue_total_observed_wait_s": 1,
        "destination_kv_queue_p95_observed_wait_s": pytest.approx(0.95),
    }


def test_agentic_shared_transfers_finish_without_floating_stall():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    full = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    workload = replace(full, records=full.records[:1])
    scenario, routes = experiment.build_scenario(workload, model, 10, 3, 0, 10, 15)
    initial = source_power(scenario, model)
    minimum = source_power(
        scenario, model, (session.session_id for session in scenario.sessions)
    )
    scenario = replace(scenario, power_limit_w=(initial + minimum) / 2)
    planned = experiment.plan(scenario, model, routes, "greedy", seed=3)
    result = experiment.execute(scenario, model, planned.moves)
    assert result.completed_sessions == len(planned.moves)


def test_small_run_reuses_plans_and_writes_raw_tables_and_plots(tmp_path: Path):
    runs = list(experiment.run(
        workload_paths=(experiment.DEFAULT_WORKLOADS[2],), sessions=6, power_limits=(500,),
        deadlines=(5,), end_s=5, solvers=("greedy",), seed=3,
        link_bytes_per_s=1_250_000_000,
    ))
    assert len(runs) == 3
    assert all(run.plan.moves == runs[0].plan.moves for run in runs)
    assert all(run.plan.profile_case == "slower" for run in runs)
    assert all(next(e for e in run.result.events if e.event == "plan_ready").time_s == 0
               for run in runs)

    summary = experiment._summary(runs[0])
    assert summary["initial_source_power_w"] == runs[0].plan.initial_source_power_w
    assert summary["kv_capacity_tokens_per_instance"] == runs[0].plan.kv_capacity_tokens
    assert summary["max_source_resident_kv_tokens"] <= summary["kv_capacity_tokens_per_instance"]
    assert summary["requested_source_drop_w"] == pytest.approx(
        runs[0].plan.initial_source_power_w - runs[0].scenario.power_limit_w
    )
    assert summary["modeled_source_drop_at_deadline_w"] == pytest.approx(
        runs[0].plan.initial_source_power_w
        - runs[0].result.modeled_source_power_at_deadline_w
    )
    assert summary["planning_profile_case"] == "slower"
    assert summary["planned_service_debt_replica_s"] == runs[0].plan.service_debt_replica_s
    assert summary["required_service_recovery_s"] == runs[0].plan.required_recovery_s
    assert summary["binding_resources"] == "|".join(runs[0].plan.binding_resources)
    assert summary["deadline_met"] == runs[0].result.deadline_met
    assert summary["migration_makespan_s"] == runs[0].result.migration_makespan_s
    assert summary["final_state_ready_s"] == runs[0].result.final_state_ready_s
    assert summary["source_sites"] == summary["destination_sites"] == 1

    experiment.write(iter(runs), tmp_path)
    for name in ("summary.csv", "events.csv", "sessions.csv", "requests.csv", "network.csv",
                 "queues.csv",
                 "power.csv", "plans.csv", "power_timeline.png", "session_pause.png",
                 "network_time.png", "request_wait.png", "expected_vs_modeled_power.png",
                 "policy_outcomes.png"):
        assert (tmp_path / name).exists()
    queue_rows = list(csv.DictReader((tmp_path / "queues.csv").open()))
    expected = {
        (run.run_id, row.session_id, row.phase):
        (row, (row.start_s if row.start_s is not None else run.scenario.end_s) - row.arrival_s)
        for run in runs for row in run.result.queues
    }
    assert len(queue_rows) == len(expected)
    for row in queue_rows:
        queue, wait = expected[row["run_id"], row["session_id"], row["phase"]]
        assert float(row["observed_wait_s"]) == pytest.approx(wait)
        assert int(row["depth_at_arrival"]) == queue.depth_at_arrival
        assert int(row["bytes_at_arrival"]) == queue.bytes_at_arrival
        assert row["pending_at_end"] == str(queue.start_s is None)


def test_write_handles_a_plan_with_no_moves(tmp_path: Path):
    runs = list(experiment.run(
        workload_paths=(experiment.DEFAULT_WORKLOADS[2],), sessions=1,
        power_limits=(10_000,), deadlines=(5,), end_s=5, solvers=("greedy",),
    ))
    assert all(not run.plan.moves for run in runs)
    experiment.write(iter(runs), tmp_path)
    assert (tmp_path / "session_pause.png").exists()


def test_worker_processes_preserve_scenario_order_and_results():
    args = {
        "workload_paths": (experiment.DEFAULT_WORKLOADS[2],),
        "sessions": 6,
        "power_limits": (500,),
        "deadlines": (5,),
        "end_s": 5,
        "solvers": ("random", "greedy"),
        "seed": 3,
    }
    serial = list(experiment.run(**args))
    parallel = list(experiment.run(**args, workers=2))

    assert [run.run_id for run in parallel] == [run.run_id for run in serial]
    for expected, actual in zip(serial, parallel):
        assert actual.scenario == expected.scenario
        assert replace(actual.plan, solve_s=0) == replace(expected.plan, solve_s=0)
        assert actual.result == expected.result
