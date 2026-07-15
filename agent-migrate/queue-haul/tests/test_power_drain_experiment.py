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

import pytest

import power_drain_experiment as experiment
from planner import source_power
from profiles import ModelProfile, WorkloadProfile


def test_build_scenario_packs_calibrated_sessions_and_named_links():
    model = ModelProfile.load(experiment.DEFAULT_MODEL)
    workload = WorkloadProfile.load(experiment.DEFAULT_WORKLOADS[2])
    scenario, route = experiment.build_scenario(workload, model, 12, 0, 500, 5, 5)

    assert len(scenario.sessions) == 12
    assert source_power(scenario, model) > 500
    assert all(len(route(f"source-{i}", "dest-0")) == 2
               for i in range(len(scenario.instances) // 2))


def test_excess_energy_integrates_step_power_after_deadline():
    power = ((0, 100, 0), (6, 40, 0), (8, 20, 0))
    assert experiment.excess_energy(power, 5, 10, 30) == pytest.approx(90)


def test_small_run_reuses_plans_and_writes_raw_tables_and_plots(tmp_path: Path):
    runs = experiment.run(
        workload_paths=(experiment.DEFAULT_WORKLOADS[2],), sessions=6, power_limits=(500,),
        deadlines=(5,), end_s=5, solvers=("load_only",),
    )
    assert len(runs) == 3
    assert all(run.plan.moves == runs[0].plan.moves for run in runs)

    experiment.write(runs, tmp_path)
    for name in ("summary.csv", "events.csv", "sessions.csv", "network.csv", "power.csv",
                 "plans.csv", "power_timeline.png", "session_pause.png", "network_time.png",
                 "policy_outcomes.png"):
        assert (tmp_path / name).exists()
