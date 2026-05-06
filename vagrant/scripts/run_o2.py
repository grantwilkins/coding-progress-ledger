"""Emit O2 oracle-vs-policy plan-diff artifacts.

For each of the four K9 diagnostic cells, compute oracle / strong reuse
/ mixed_min_pressure / random_mode plans, simulate them through K4,
and write a per-workflow plan-diff plus per-policy bottleneck breakdown.

The four cells (matched to `scripts/run_k8_k9.py`):

    tiny_prefill_pressure          n=4, tiny,     tight prefill, 100 Gbps
    medium_multi_resource          n=4, medium,   tight prefill,   5 Gbps
    monorepo_workspace_pressure    n=4, monorepo, loose prefill, 100 Gbps
    slow_link_network_pressure     n=4, medium,   loose prefill,   1 Gbps
"""
from __future__ import annotations

from pathlib import Path

from vagrant_agent.k8_regime import (
    RegimeCell,
    default_bundle,
    make_k8_budget,
    make_k8_episode,
)
from vagrant_agent.oracle_diff import (
    OracleDiffReport,
    compute_oracle_diff,
    write_oracle_diff_artifacts,
)


REPO = Path(__file__).resolve().parents[1]


O2_SCENARIOS: tuple[tuple[str, str, str, int], ...] = (
    ("tiny_prefill_pressure",       "tiny",     "tight", 100),
    ("medium_multi_resource",       "medium",   "tight",   5),
    ("monorepo_workspace_pressure", "monorepo", "loose", 100),
    ("slow_link_network_pressure",  "medium",   "loose",   1),
)


def main() -> None:
    bundle = default_bundle(REPO)
    reports: list[OracleDiffReport] = []
    for i, (scenario, state_scale, prefill_capacity, link_gbps) in enumerate(O2_SCENARIOS):
        cell = RegimeCell(
            n_workflows=4,
            state_scale=state_scale,
            prefill_capacity=prefill_capacity,
            link_gbps=link_gbps,
            seed=9009 + i,
        )
        episode, manifests = make_k8_episode(cell)
        budget = make_k8_budget(cell)
        report = compute_oracle_diff(
            scenario=scenario,
            cell={
                "n_workflows": cell.n_workflows,
                "state_scale": state_scale,
                "prefill_capacity": prefill_capacity,
                "link_gbps": link_gbps,
            },
            episode=episode,
            manifests=manifests,
            bundle=bundle,
            budget=budget,
            random_seed=cell.seed,
        )
        reports.append(report)
    write_oracle_diff_artifacts(reports, REPO / "runs" / "o2_oracle_diff")


if __name__ == "__main__":
    main()
