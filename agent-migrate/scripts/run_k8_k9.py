"""Emit K8 regime_map and K9 oracle artifacts.

Default run is intentionally bounded so it can be refreshed during normal
development.  The reusable K8 module supports the full 4 x 5 x 3 x 4 sweep;
this script emits a representative first pass over the same axes.
"""
from __future__ import annotations

from pathlib import Path

from agent_migrate_agent.k8_regime import (
    RegimeCell,
    calibrate_k8_estimator,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
    run_k8_sweep,
    write_k8_calibration_artifacts,
    write_k8_artifacts,
)
from agent_migrate_agent.k9_oracle import (
    OracleScenarioResult,
    run_small_n_oracle,
    write_oracle_sweep_artifacts,
)


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    bundle = default_bundle(REPO)
    k8_rows = run_k8_sweep(
        bundle,
        n_values=(10, 100, 1_000, 10_000),
        state_scales=("tiny", "swe_bench", "medium", "monorepo", "large_artifact"),
        prefill_caps=("loose", "moderate", "tight"),
        link_gbps_values=(1, 5, 25, 100),
        exact=False,
    )
    write_k8_artifacts(k8_rows, REPO / "runs" / "k8_regime_map")
    calibration_rows = calibrate_k8_estimator(bundle)
    write_k8_calibration_artifacts(calibration_rows, REPO / "runs" / "k8_regime_map")

    oracle_scenarios = (
        ("tiny_prefill_pressure", "tiny", "tight", 100),
        ("medium_multi_resource", "medium", "tight", 5),
        ("monorepo_workspace_pressure", "monorepo", "loose", 100),
        ("slow_link_network_pressure", "medium", "loose", 1),
    )
    oracle_rows: list[OracleScenarioResult] = []
    for i, (scenario, state_scale, prefill_capacity, link_gbps) in enumerate(oracle_scenarios):
        k9_cell = RegimeCell(
            n_workflows=4,
            state_scale=state_scale,
            prefill_capacity=prefill_capacity,
            link_gbps=link_gbps,
            seed=9009 + i,
        )
        episode, manifests = make_k8_episode(k9_cell)
        oracle_rows.append(OracleScenarioResult(
            scenario=scenario,
            state_scale=state_scale,
            prefill_capacity=prefill_capacity,
            link_gbps=link_gbps,
            result=run_small_n_oracle(
                episode, manifests, bundle, make_k8_budget(k9_cell), max_workflows=4,
            ),
        ))
    write_oracle_sweep_artifacts(oracle_rows, REPO / "runs" / "k9_oracle")


if __name__ == "__main__":
    main()
