"""
Claim:
The experiment samples complete sessions, uses absolute local power limits, plans
once centrally, and preserves raw execution evidence for every uncertainty case.

Plausible wrong implementations:
- Resample independent workload fields and create impossible sessions.
- Re-plan the faster/slower cases and hide plan sensitivity.
- Test the last power sample instead of integrating the deadline window.
- Emit only a summary and make timing or network claims impossible to audit.
"""

from pathlib import Path
import math
from dataclasses import replace

import pytest

import power_drain_experiment as experiment
from planner import source_power
from profiles import ModelProfile, WorkloadProfile


def test_build_scenario_packs_calibrated_sessions_and_named_links():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    scenario, route = experiment.build_scenario(workload, model, 12, 3, 500, 5, 5)

    assert len(scenario.sessions) == 12
    assert source_power(scenario, model) > 500
    assert all(len(route(f"source-{i}", "dest-0")) == 2
               for i in range(len(scenario.instances) // 2))


def test_expected_load_and_wake_probability_match_sampled_request_timing():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    records = workload.sample(6, 3)
    scenario, _ = experiment.build_scenario(
        workload, model, 6, 3, 500, 5, 5, controller_delay_s=1
    )
    for session, record in zip(scenario.sessions, records):
        cycle = record.request_gap_s + record.tool_delay_s
        assert session.expected_f == pytest.approx(record.prompt_tokens / cycle)
        expected = 1 - math.exp(-(4 - record.tool_delay_s) / record.request_gap_s)
        assert session.wake_probability == pytest.approx(expected)


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


def test_small_run_reuses_plans_and_writes_raw_tables_and_plots(tmp_path: Path):
    runs = list(experiment.run(
        workload_paths=(experiment.DEFAULT_WORKLOADS[2],), sessions=6, power_limits=(500,),
        deadlines=(5,), end_s=5, solvers=("load_only",), seed=3,
    ))
    assert len(runs) == 3
    assert all(run.plan.moves == runs[0].plan.moves for run in runs)
    assert all(next(e for e in run.result.events if e.event == "plan_ready").time_s == 0
               for run in runs)

    experiment.write(iter(runs), tmp_path)
    for name in ("summary.csv", "events.csv", "sessions.csv", "requests.csv", "network.csv",
                 "power.csv", "plans.csv", "power_timeline.png", "session_pause.png",
                 "network_time.png", "request_wait.png", "expected_vs_modeled_power.png",
                 "policy_outcomes.png"):
        assert (tmp_path / name).exists()
